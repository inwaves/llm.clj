(defproject llm.clj "0.1.0-SNAPSHOT"
  :description "LLM training in Clojure - a port of llm.c with Neanderthal acceleration"
  :url "https://github.com/inwaves/llm.clj"
  :license {:name "MIT"
            :url "https://opensource.org/licenses/MIT"}
  
  :dependencies [[org.clojure/clojure "1.11.1"]
                 ;; High-performance linear algebra with OpenBLAS backend
                 [org.uncomplicate/neanderthal-base "0.57.0"]
                 [org.uncomplicate/neanderthal-openblas "0.57.0"]
                 ;; CUDA backend for GPU acceleration
                 [org.uncomplicate/neanderthal-cuda "0.57.0"]
                 ;; CUDA support for GPU acceleration
                 [uncomplicate/clojurecuda "0.16.0"]
                 ;; Performance benchmarking
                 [criterium "0.4.6"]]
  
  :repl-options {:init-ns llm.train-gpt2}
  
  :profiles {:dev {:dependencies [[org.clojure/test.check "1.1.1"]]}}
  
  ;; Java options for Neanderthal native libraries
  :jvm-opts ["-Dclojure.compiler.direct-linking=true"
             "-XX:MaxDirectMemorySize=16g"
             "-XX:+UseLargePages"]
  
  :source-paths ["src"]
  :test-paths ["test"]
  
  :aliases {"bench" ["run" "-m" "llm.neo.benchmark"]})