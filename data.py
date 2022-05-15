
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import re

class Sequence:
    def __init__(self, header, seq):
        self.seq = seq
        self.header = header
        self.meta = header[header.rfind(':: ') + 3:]
        self.meta = self.meta.split(' ')

        self.pct_id = float(self.meta[0])

        self.q_start = int(self.meta[1])
        self.q_end = int(self.meta[2])
        self.t_start = int(self.meta[3])
        self.t_end = int(self.meta[4])

        self.start = self.t_start
        self.end = self.t_end

        self.cluster = self.meta[5]

    def write_to_file(self, file):
        f = open(file, 'w')
        f.write(self.header + "\n")
        f.write(self.seq + "\n")
        f.close()

    def random_subrange(self, window_size):
        if (self.end - self.start) > window_size:
            diff = (self.end - self.start) - window_size
            self.start += np.random.randint(0, diff)
            self.end = self.start + window_size


def read_posterior_file(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        found_dump = False
        read_line_count = 0
        for line in file:
            if not found_dump:
                if "POSTERIOR DUMP" in line:
                    found_dump = True
                    continue

            else:
                row = []
                if "END DUMP" in line:
                    break
                if read_line_count >= 2:
                    line = line.strip()
                    line = re.split(" +", line)
                    if len(line) == 1:
                        continue

                    if line[1] != 'M':
                        continue
                    for i in range(2, len(line)):
                        if line[i] == '-inf':
                            val = -100
                        else:
                            val = float(line[i])

                        row.append(val)

                    matrix.append(row)
                read_line_count += 1

    return torch.tensor(matrix)  # [:,:-5]

class SequenceDB:
    def __init__(self, file_path_A, file_path_B):

        if file_path_A is None and file_path_B is None:
            self.A_seqs = []
            self.B_seqs = []
            self.matrices = []
            return


        self.A_seqs = []
        self.B_seqs = []
        self.matrices = []

        file_A = open(file_path_A, 'r')
        file_B = open(file_path_B, 'r')

        lines_A = file_A.readlines()
        lines_B = file_B.readlines()

        if len(lines_A) != len(lines_B):
            print("ERROR: File A and B have different number of lines ", len(lines_A), len(lines_B))
            exit()

        headerA = ""
        seqA = ""
        headerB = ""
        seqB = ""

        for i in range(len(lines_A)):
            if i % 2 == 0:
                if headerA != "":
                    self.A_seqs.append(Sequence(headerA, seqA))
                    self.B_seqs.append(Sequence(headerB, seqB))

                    seqA = ""
                    seqB = ""

                headerA = lines_A[i].strip()
                headerB = lines_B[i].strip()

            else:
                seqA = lines_A[i].strip()
                seqB = lines_B[i].strip()

        file_A.close()
        file_B.close()


    def correct_sequence_ranges(self, min_len = 50):
        A_seqs = []
        B_seqs = []

        for i in range(len(self.A_seqs)):
            s1 = self.A_seqs[i]
            s2 = self.B_seqs[i]

            q_start = max(s1.q_start, s2.q_start)
            q_end = min(s1.q_end, s2.q_end)

            s1.start += q_start - s1.q_start
            s2.start += q_start - s2.q_start

            s1.end -= s1.q_end - q_end
            s2.end -= s2.q_end - q_end

            if s1.end - s1.start > min_len:
                if s2.end - s2.start > min_len:
                    A_seqs.append(s1)
                    B_seqs.append(s2)

        self.A_seqs = A_seqs
        self.B_seqs = B_seqs

    def random_seqdb(self, num_sequences):
        seqs = np.random.choice(len(self.A_seqs), size=num_sequences, replace=False)

        newSeqDB = SequenceDB(None, None)

        for i in seqs:
            newSeqDB.A_seqs.append(self.A_seqs[i])
            newSeqDB.B_seqs.append(self.B_seqs[i])

        return newSeqDB

    def output_to_dir(self, dir, pdump=None):

        for i in range(len(self.A_seqs)):
            self.A_seqs[i].write_to_file(dir + "/A_" + str(i) + ".fa")
            self.B_seqs[i].write_to_file(dir + "/B_" + str(i) + ".fa")

        if pdump != None:
            file = open(dir + "/" + "pdump.sh", 'w')
            for i in range(len(self.A_seqs)):
                file.write(pdump + " " + dir + "/A_" + str(i) + ".fa " + dir + "/B_" + str(i) + ".fa > " + dir + "/P_" + str(i) + ".txt\n")
            file.write("echo done > " + dir + "/done.done\n")
            file.close()

    def read_posteriors(self, dir):
        for i in range(len(self.A_seqs)):
            self.matrices.append(read_posterior_file(dir + "/P_" + str(i) + ".txt"))

    def posteriors_to_targets(self):
        for i in range(len(self.matrices)):
            s1 = self.A_seqs[i]
            s2 = self.B_seqs[i]
            mat = self.matrices[i][s2.start:s2.end, s1.start:s1.end]

            mean = torch.mean(mat, dim=1)
            std = torch.std(mat, dim=1)

            mat = (mat.T - mean).T
            mat = (mat.T / std).T
            mat = mat - 2.0

            self.matrices[i] = torch.sigmoid(mat)

    def posteriors_to_probabilities(self, pivot=-25, mul=0.05):
        for i in range(len(self.matrices)):
            s1 = self.A_seqs[i]
            s2 = self.B_seqs[i]
            mat = self.matrices[i][s2.start:s2.end, s1.start:s1.end]
            mat[mat > pivot] *= mul
            self.matrices[i] = mat

amino_n_to_a = [c for c in 'ARNDCQEGHILKMFPSTWYVBZXJ*']
amino_a_to_n = {c: i for i, c in enumerate('ARNDCQEGHILKMFPSTWYVBZXJ*')}
amino_frequencies = torch.tensor([0.074,
                                  0.042,
                                  0.044,
                                  0.059,
                                  0.033,
                                  0.058,
                                  0.037,
                                  0.074,
                                  0.029,
                                  0.038,
                                  0.076,
                                  0.072,
                                  0.018,
                                  0.040,
                                  0.050,
                                  0.081,
                                  0.062,
                                  0.013,
                                  0.033,
                                  0.068])


amino_n_to_v = torch.zeros(len(amino_n_to_a), 20)
for i in range(20):
    amino_n_to_v[i,i] = 1.0

amino_n_to_v[amino_a_to_n['B'],amino_a_to_n['D']] = 0.5
amino_n_to_v[amino_a_to_n['B'],amino_a_to_n['N']] = 0.5

amino_n_to_v[amino_a_to_n['Z'],amino_a_to_n['Q']] = 0.5
amino_n_to_v[amino_a_to_n['Z'],amino_a_to_n['E']] = 0.5

amino_n_to_v[amino_a_to_n['J'],amino_a_to_n['I']] = 0.5
amino_n_to_v[amino_a_to_n['J'],amino_a_to_n['L']] = 0.5

amino_n_to_v[amino_a_to_n['X']] = amino_frequencies
amino_n_to_v[amino_a_to_n['*']] = amino_frequencies