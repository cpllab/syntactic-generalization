

def checkpoints_to_reference = [
    "tinylstm": "docker://cpllab/language-models:tinylstm",
]

checkpoints = Channel.fromPath("checkpoints/*/*", type: "dir").map { f ->
    // yields (path, (model, corpus, seed)) tuples
    def match = (f.toString() =~ /checkpoints\/(.+)\/(.+)_(.+)$/)[0]
    tuple(f, match[1], match[2], match[3])
}


suites = Channel.fromPath("test_suites/json/*.json")


process computeSuiteSurprisals {
    conda "/home/jon/anaconda3"
    input:
    set file(checkpoint_dir), val(model_name), corpus, seed, file(test_suite) \
        from checkpoints.combine(suites)

    output:
    set model_name, corpus, seed, test_suite, file("results.json") \
        into run_results

    script:
    model_ref = checkpoints_to_reference[model_name]

    """
    /usr/bin/env bash
    export PATH=/home/jon/Projects/syntaxgym/cli/bin:/home/jon/anaconda3/bin
    export PYTHONPATH=/home/jon/Projects/syntaxgym/cli:/home/jon/Projects/lm-zoo
    pushd /home/jon/Projects/syntaxgym/cli && pipenv shell && pushd
    syntaxgym compute-surprisals ${model_ref} ${test_suite} > results.json
    """
}
