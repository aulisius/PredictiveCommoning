#include "llvm/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OrderedBasicBlock.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
// Defines the value to used during debugging
#define DEBUG_TYPE "predictive"

// Defines the type of indexed expression. Possible values are {RHS, LHS}
#define IE_TYPE "loop.type"

// Defines the constant which replaces the induction variable
#define IE_ITER "loop.iter"

// Defines the array affected by the transformation
#define IE_ARRAY "loop.array"

// Marks the instruction which will be useless after transformation
#define IE_DEAD "loop.dead"

using namespace llvm;

bool isLHS(Instruction *i) {
  return std::any_of(
    i->user_begin(),
    i->user_end(),
    [](User *user) { return isa<StoreInst>(user); });
}

SmallVector<Instruction*, 4> getGEPs(BasicBlock *BB) {
  SmallVector<Instruction*, 4> onlyGEPs;
  std::for_each(
    BB->begin(),
    BB->end(),
    [&](Instruction &i) {
      if (isa<GetElementPtrInst>(i)) {
        onlyGEPs.push_back(&i);
      }
    });
  return onlyGEPs;
}

bool containsLHS(BasicBlock *BB) {
  auto onlyGEPs = getGEPs(BB);
  return std::any_of(onlyGEPs.begin(), onlyGEPs.end(), isLHS);
}

namespace {
  class Pass1 : public LoopPass {
  public:
    static char ID;
    Pass1() : LoopPass(ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
      getLoopAnalysisUsage(AU);
    }

    int getInitial(Instruction *index, Instruction *arrayptr) {
      for (auto tmpL = index, tmpA = arrayptr; tmpL != tmpA;) {
        if(auto bi = dyn_cast<BinaryOperator>(tmpL)) {
          for (auto l = bi->op_begin(); l != bi->op_end(); ++l) {
            if (auto c = dyn_cast<ConstantInt>(l)) {
              return c->getSExtValue();
            }
          }
        }
        if (tmpL->user_empty()) break;
        tmpL = tmpL->user_back();
      }
      return 0;
    }

    void setUsersMetadata(Instruction *start, Instruction *end, MDNode *node) {
      for (auto tmpL = start, tmpA = end; tmpL != tmpA;) {
        tmpL->setMetadata(IE_DEAD, node);
        if (tmpL->user_empty()) break;
        tmpL = tmpL->user_back();
      }
    }

    Constant* makeConstant(LLVMContext& context, int value) {
      return ConstantInt::get(Type::getInt32Ty(context), value);
    }

    Instruction* findDominantLoad(BasicBlock *BB, Instruction *GEP) {
      OrderedBasicBlock oBB(BB);
      Instruction *dominant = nullptr;
      for (auto it = BB->begin(); it != BB->end(); ++it) {
        if (oBB.dominates(&*it, GEP) && isa<LoadInst>(it)) {
          dominant = &*it;
        }
      }
      return dominant;
    }

    Instruction* findDominatedLoad(Instruction *GEP) {
      return GEP->user_empty()
        ? nullptr
        : GEP->user_back();
    }

    void setAllMetadata(BasicBlock *BB) {
      LLVMContext& context = BB->getContext();
      MDBuilder builder(context);
      auto *RHS = MDNode::get(context, builder.createString("RHS"));
      auto *LHS = MDNode::get(context, builder.createString("LHS"));
      auto *deadInst = MDNode::get(context, builder.createString(""));

      auto onlyGEPs = getGEPs(BB);
      for (auto gep = onlyGEPs.begin(); gep != onlyGEPs.end(); gep++) {
        auto gepIT = *gep;

        auto *dominant = findDominantLoad(BB, gepIT);
        auto *dominated = findDominatedLoad(gepIT);

        if (dominant && dominated) {
          if (isa<LoadInst>(dominated)) {
            dominant->setMetadata(IE_DEAD, deadInst);
            gepIT->setMetadata(IE_TYPE, RHS);
            gepIT->setMetadata(IE_DEAD, deadInst);
            setUsersMetadata(dominant, gepIT, deadInst);
            auto cnt = getInitial(dominant, gepIT);
            auto iterV = makeConstant(context, cnt);
            auto iterMD = MDNode::get(context, builder.createConstant(iterV));
            dominated->setMetadata(IE_ITER, iterMD);
            dominated->setMetadata(IE_DEAD, deadInst);
          } else {
            gepIT->setMetadata(IE_TYPE, LHS);
          }
        }
      }
    }

    bool runOnLoop(Loop *loop, LPPassManager &LPM) override {
      for (auto block : loop->getBlocks()) {
        if (containsLHS(block)) {
          setAllMetadata(block);
        }
      }
      return true;
    }
  };
}

char Pass1::ID = 0;
static RegisterPass<Pass1> F(
  "pass1",
  "Predictive Commoning"
);

namespace {
  struct Pass2 : public FunctionPass, public InstVisitor<Pass2> {
  public:
    static char ID;
    Pass2() : FunctionPass(ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
    }

    Constant* makeConstant(LLVMContext& context, int value) {
      return ConstantInt::get(Type::getInt32Ty(context), value);
    }

    bool runOnFunction(Function &F) override {
      SmallString<10> fname(F.getName());
      if (fname.startswith("_")) {
        return false;
      }

      LoopInfo& LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
      LLVMContext& ctx = F.getContext();
      auto entry = &F.getEntryBlock();
      auto inst = entry->getSingleSuccessor();
      IRBuilder<> builder(entry->getTerminator());

      for (auto loops = LI.begin(); loops != LI.end(); loops++) {
        bool isPCDone = false;

        Loop *L = *loops;
        BasicBlock *lbody = nullptr;
        SmallVector<AllocaInst*, 4> newAllocs;
        SmallVector<Instruction*, 4> tmpStores, oldStores, oldLoads;

        for (auto it = L->block_begin(); it != L->block_end(); ++it) {
          BasicBlock *BB = *it;
          if (containsLHS(BB)) {
            lbody = BB;
          }
        }
        if (lbody == nullptr) {
          continue;
        }
        auto onlyGEPs = getGEPs(lbody);
        for (auto it = onlyGEPs.begin(); it != onlyGEPs.end(); it++) {
          auto *inst = *it;
          if (auto md = inst->getMetadata(IE_TYPE)) {
            isPCDone = true;
            auto p = builder.CreateAlloca(Type::getInt32Ty(ctx), nullptr, "p");
            p->setAlignment(4);

            auto type = cast<MDString>(md->getOperand(0))->getString();
            if (type == "LHS") {
              tmpStores.push_back(p);
              oldStores.push_back(inst->user_back());
            } else {
              newAllocs.push_back(p);
              auto oldLoad = inst->user_back();
              oldLoads.push_back(oldLoad);

              auto MD = oldLoad->getMetadata(IE_ITER);
              int index = mdconst::extract<ConstantInt>(MD->getOperand(0))->getSExtValue();

              auto cnt = makeConstant(ctx, 0);
              auto arrindex = makeConstant(ctx, index);
              auto gepI = builder.CreateInBoundsGEP(inst->getOperand(0), { cnt, arrindex });
              auto lI = builder.CreateAlignedLoad(gepI, 4);
              builder.CreateAlignedStore(lI, p, 4);
            }
          }
        }
        IRBuilder<> loopBuilder(lbody, lbody->getFirstInsertionPt());
        int i = 0;
        for (auto it : newAllocs) {
          auto newLoad = loopBuilder.CreateAlignedLoad(it, 4);
          oldLoads[i]->replaceAllUsesWith(newLoad);
          i++;
        }
        i = 0;
        for (auto store: oldStores) {
          auto *user = cast<Instruction>(store->getOperand(0));
          IRBuilder<> tmpLoad(store->getNextNode());
          tmpLoad.CreateAlignedStore(user, tmpStores[i], 4);
          for (auto it = newAllocs.rbegin(); it != newAllocs.rend(); ++it) {
            auto prevInst = tmpLoad.CreateAlignedLoad(*it, 4);
            tmpLoad.CreateAlignedStore(prevInst, *(++it), 4);
          }
          auto prevInst = tmpLoad.CreateAlignedLoad(tmpStores[i], 4);
          tmpLoad.CreateAlignedStore(prevInst, *newAllocs.rbegin(), 4);
          i++;
        }
      }
      return true;
    }
  };
}

char Pass2::ID = 0;
static RegisterPass<Pass2> Y(
  "pass2",
  "PredictiveCommoning pass 2"
);

namespace {
  struct Pass3 : public LoopPass {
  public:
    static char ID;
    Pass3() : LoopPass(ID) {}

    bool runOnLoop(Loop *L, LPPassManager &LPM) override {
      for (auto block : L->getBlocks()) {
        for (auto i = block->rbegin(); i != block->rend(); i++) {
          if(i->getMetadata(IE_DEAD)) {
            i->dropUnknownNonDebugMetadata();
            i->replaceAllUsesWith(UndefValue::get(i->getType()));
            i->eraseFromParent();
          }
        }
      }
      return true;
    }
  };
}

char Pass3::ID = 0;
static RegisterPass<Pass3> P(
  "pass3",
  "PredictiveCommoning pass 3"
);

