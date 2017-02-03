import re
import os
import pickle


dir_raw_text = "/Users/haohanchen/Dropbox/WORK/Research (now)/NN-vote/Data"


def clean_string(string, clean = True):
	if clean:
		string = re.sub("A BILL", "", string)
		string = re.sub("AN ACT", "", string)
		strin = re.sub("IN THE HOUSE OF REPRESENTATIVES", "", string)
		string = re.sub("all", "", string)
		string = re.sub("ENDENDEND", "", string)
		string = re.sub(r"\((\d+|\w+)\)", " ", string)
		string = re.sub("\r\n", " ", string)
		string = re.sub(r"[^A-Za-z0-9().,!?]", " ", string)     
		string = re.sub(r"^\([a-z]\)", " ",string) 
		string = re.sub(r"\'s", " \'s", string) 
		string = re.sub(r"\'ve", " \'ve", string) 
		string = re.sub(r"n\'t", " n\'t", string) 
		string = re.sub(r"\'re", " \'re", string) 
		string = re.sub(r"\'d", " \'d", string) 
		string = re.sub(r"\'ll", " \'ll", string) 

		string = re.sub(r",", " , ", string) 
		string = re.sub(r"\.[^\d\w]", " . ", string) 
		string = re.sub(r"\:", " : ", string) 
		string = re.sub(r"\;", " ; ", string) 
		string = re.sub(r"\!", " !", string) 
		string = re.sub(r"\(", " ( ", string) 
		string = re.sub(r"\)", " ) ", string) 
		string = re.sub(r"\? ", " ?", string) 
		string = re.sub(r"\-+", " -- ", string) 
		string = re.sub(r" +", " ",string) 	
		string = string.rstrip().lstrip()
	return string


Bills = dict(HR = [], date = [], participants = [], summary = [], main = [])

with open(os.path.join(dir_raw_text, 'billtext110raw.txt')) as raw_bills:
	write = 0
	date = ""
	main = ""
	summary = ""
	participants = ""
	num_bill = 0
	for line in raw_bills.readlines():
		line = line.lstrip()
		# line = next(raw_bills).lstrip()
		if line == "":
			pass


		# Get Bill summary
		if write == 1:
			summary = summary + line

		# Get Bill date
		if write == 2:
			if bool(re.match(r"^(January|February|March|April|May|June|July|August|September|October|November|December)", line)):
				date = line
			else:
				participants = participants + line

		# Get Bill main text
		if write == 3:
			main = main + line

		# Get BIll HR number
		if bool(re.match(r"^H. R.", line)):
			HR = line.split()[2]
			#print HR
			Bills['HR'].append(HR)
			write = write + 1
		
		if bool(re.match(r"^_", line)):
			write = write + 1

		if re.match(r"^ENDENDEND", line):
			num_bill = num_bill + 1
			print "Number of Bills: ", num_bill

			Bills["main"].append(clean_string(main))
			Bills["summary"].append(clean_string(summary))
			Bills["participants"].append(clean_string(participants))
			Bills["date"].append(clean_string(date))
			write = 0
			summary = ""
			main = ""
			participants = ""


pickle.dump(Bills, open("/Users/haohanchen/Dropbox/WORK/Research (now)/NN-vote/Data/house110.p", "wb"))
		


# Low cap. space before stop, comma etc.
# Transfer to numpy & pickle, save
# remove (1), (2)...
# remove numbers