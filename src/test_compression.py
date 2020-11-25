import os
from taggers import bilstm_aux, bilstm_crf, svmtool, stanford, meta_tagger
from taggers import flair_pos, bert_bpemb
from taggers import nltk_tnt, nltk_crf, nltk_brill, nltk_hmm
from util.model_compression import shannon_entropy, COMPRESSION_EXTS

def test_compression_methods(tagger):
    formats = ["zip", "gztar", "bztar", "xztar"]
    results = []
    entropy_before = shannon_entropy(*tagger.necessary_model_files())
    size_before = tagger.model_size()
    print(f"Entropy before: {entropy_before}")
    print(f"Size before: {size_before} KB")

    for comp_format in formats:
        print("Compressing...")
        compressed = tagger.compress_model(comp_format)
        entropy_after = shannon_entropy(compressed)
        size_after = tagger.compressed_model_size(comp_format)
        results.append(
            (comp_format, entropy_before, entropy_after, size_before, size_after)
        )
        print(f"Entropy after: {entropy_after}")
        print(f"Size after: {size_after} KB")
        os.remove(compressed)

    return results

if __name__ == "__main__":
    TAGGERS = {
        "bilstm_aux": bilstm_aux.BILSTMAUX,
        "bilstm_crf": bilstm_crf.BILSTMCRF,
        "svmtool": svmtool.SVMT,
        "stanford": stanford.Stanford,
        "tnt": nltk_tnt.TnT,
        "brill": nltk_brill.Brill,
        "crf": nltk_crf.CRF,
        "hmm": nltk_hmm.HMM,
        "meta_tagger": meta_tagger.METATAGGER,
        "flair": flair_pos.Flair
        # bert_bpemb": bert_bpemb.BERT_BPEMB,
    }

    class Args:
        def __init__(self):
            self.lang = "en"
            self.treebank = "gum"
            self.iter = 10

    with open("compression_test.txt", "w", encoding="utf-8") as file_out:
        for tagger in TAGGERS:
            print(f"########## {tagger} ##########")
            test_results = test_compression_methods(TAGGERS[tagger](Args(), tagger))
            file_out.write(tagger + "\n")
            for comp_format, ent_before, ent_after, size_before, size_after in test_results:
                file_out.write(
                    f"{comp_format} {ent_before} {ent_after} {ent_after - ent_before} " +
                    f"{ent_after / ent_before} {size_before} {size_after} " +
                    f"{size_before - size_after} {size_before / size_after}\n"
                )

            file_out.write("\n")
