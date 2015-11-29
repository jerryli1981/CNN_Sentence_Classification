
import random
import re
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

if __name__ == "__main__":

	pos = "rt-polarity.pos"
	neg = "rt-polarity.neg"

	li = []
	with open(pos, 'r') as f1, open(neg, 'r') as f2:
		 for pos_s, neg_s in zip(f1, f2):
		 	pos_s = clean_str(pos_s)
		 	neg_s = clean_str(neg_s)
		 	pos_s = pos_s.rstrip('\n')
		 	neg_s = neg_s.rstrip('\n')
		 	pos_s += "\t1"
		 	neg_s += "\t0"
		 	li.append(pos_s)
		 	li.append(neg_s)

	random.shuffle(li)
	len_li = len(li)
	print "total sent is, ", len_li

	train_size = 0
	dev_size = 0
	with open("train.txt", 'w') as train, open("dev.txt", 'w') as dev, open("test.txt" , 'w') as test:
		for sent in li[len_li/2:]:
			test.write(sent+"\n")

		for sent in li[:len_li/2]:
			if random.randint(1, 10) == 5:
				dev.write(sent+"\n")
				dev_size += 1
			else:
				train.write(sent+"\n")
				train_size += 1

	print "train_size %s, dev_size %s"%(train_size, dev_size)