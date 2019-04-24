## Global alignment to reduce pairwise similarity##

from Bio.Emboss.Applications import NeedleCommandline
import re
import os

def needle_align_code(query_seq, target_seq):
    needle_cline = NeedleCommandline(asequence="asis:" + query_seq,
                                     bsequence="asis:" + target_seq,
                                     aformat="simple",
                                     gapopen=10,
                                     gapextend=0.5,
                                     outfile='stdout'
                                     )
    out_data, err = needle_cline()
    out_split = out_data.split("\n")
    p = re.compile("\((.*)\)")
    return p.search(out_split[25]).group(1).replace("%", "")

def make_seq_dict(fasta_file):
    sequence_dict = dict()
    from Bio import SeqIO
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        sequence_dict[seq_record.id] = str(seq_record.seq)
    return sequence_dict

def make_tuple_hash(test_list,train_list,test_dict,train_dict):
    pairwise_similarity_hash = dict()
    for i in test_list:
        for j in train_list:
            pairwise_similarity_hash[(i,j)] = needle_align_code(test_dict[i],train_dict[j])
    return pairwise_similarity_hash

test_file = sys.argv[1]
train_file = sys.argv[2]
outprefix  = sys.argv[3]
test_dict = make_seq_dict(fasta_file=test_file)
train_dict = make_seq_dict(fasta_file=train_file)
test_list = list(test_dict.keys())
train_list = list(train_dict.keys())
firstfold_identity = make_tuple_hash(test_list,train_list,test_dict,train_dict)
fileX = open('Pseudouridine_hash_pairs_%d.pickle' % outprefix, 'wb')
pickle.dump(firstfold_identity, fileY, protocol=4)
fileX.close()
