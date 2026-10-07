#!/bin/sh
# Build native/lev_llm.c and native/lev_decision.cpp (with the decision
# engine of the vendored fork, tools/parallel-decision) against llama.cpp
# (native/llama.cpp, fetched by the `jolt llama` task at the pinned commit)
# into liblev_llm.{dylib,so} for jolt run/test and liblev_llm.a for jolt
# build. Run from the project root; $1 is the commit, for lev_llm_version.
set -e
L=native/llama.cpp
REV=${1:-$(cd "$L" && git rev-parse HEAD 2>/dev/null || echo unknown)}
TAG="thecodacus/parallel-decision@$(echo "$REV" | cut -c1-9)"
LIBS="$L/build/common/libllama-common.a $L/build/common/libllama-common-base.a $L/build/vendor/cpp-httplib/libcpp-httplib.a
      $L/build/src/libllama.a $L/build/ggml/src/libggml.a $L/build/ggml/src/libggml-cpu.a $L/build/ggml/src/libggml-base.a"
INC="-I$L/include -I$L/ggml/include -I$L/common -I$L/vendor -I$L/tools/parallel-decision -Inative"
CFLAGS="-O2 -std=c11 -Wall -Wextra -DLEV_LLAMA_BUILD=\"$TAG\" $INC"
CXXFLAGS="-O2 -std=c++17 -Wall $INC"
OBJS="native/lev_llm.o native/lev_decision.o native/decision-engine.o"
case "$(uname -s)" in
  Darwin)
    LIBS="$LIBS $L/build/ggml/src/ggml-metal/libggml-metal.a"
    FRAMEWORKS="-framework Metal -framework Foundation -framework MetalKit -framework Accelerate"
    cc $CFLAGS -c native/lev_llm.c -o native/lev_llm.o
    c++ $CXXFLAGS -c native/lev_decision.cpp -o native/lev_decision.o
    c++ $CXXFLAGS -w -c $L/tools/parallel-decision/decision-engine.cpp -o native/decision-engine.o
    c++ -dynamiclib $OBJS $LIBS $FRAMEWORKS -o native/liblev_llm.dylib
    # one archive with every llama.cpp member inside, for jolt build's force-load
    libtool -static -o native/liblev_llm.a $OBJS $LIBS
    # ld64 links a framework stub by lib<Name>.tbd in an -L dir: what deps.edn's
    # :static {:lib ...} entries can express
    sdk=$(xcrun --show-sdk-path)
    mkdir -p native/frameworks
    for f in Metal Foundation MetalKit Accelerate; do
      ln -sf "$sdk/System/Library/Frameworks/$f.framework/$f.tbd" "native/frameworks/lib$f.tbd"
    done
    ;;
  *)
    cc $CFLAGS -fPIC -c native/lev_llm.c -o native/lev_llm.o
    c++ $CXXFLAGS -fPIC -c native/lev_decision.cpp -o native/lev_decision.o
    c++ $CXXFLAGS -w -fPIC -c $L/tools/parallel-decision/decision-engine.cpp -o native/decision-engine.o
    c++ -shared $OBJS -Wl,--whole-archive $LIBS -Wl,--no-whole-archive -lm -lpthread -o native/liblev_llm.so
    printf 'create native/liblev_llm.a\n' > native/llm.mri
    for o in $OBJS; do printf 'addmod %s\n' "$o" >> native/llm.mri; done
    for a in $LIBS; do printf 'addlib %s\n' "$a" >> native/llm.mri; done
    printf 'save\nend\n' >> native/llm.mri
    ar -M < native/llm.mri
    rm -f native/llm.mri
    ;;
esac
rm -f $OBJS
echo "built: native/liblev_llm.* ($TAG)"
