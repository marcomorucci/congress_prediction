# Clean 112th Congress Text

import re
import os
import pickle


dir_raw_text = "/Users/haohanchen/Dropbox/WORK/Research (now)/NN-vote/Data"


def clean_string(string, clean = True):
	if clean:
		string = re.sub(r"[^A-Za-z0-9().,!?;:]", " ", string)     

		string = re.sub("A BILL", "", string)
		string = re.sub("AN ACT", "", string)
		string = re.sub("IN THE HOUSE OF REPRESENTATIVES", "", string)
		string = re.sub("IN THE SENATE OF REPRESENTATIVES", "", string)
		string = re.sub("<all>", "", string)
		string = re.sub("ENDENDEND", "", string)
		# Remove Item numbers
		string = re.sub(r"\((\d+|\w+)\)", " ",string)
		# Remove Section Title indicators
		string = re.sub(r"(SEC\.|SECTION\.|Sec\.|Section\.) \d+", "SEC_TITLE", string)

		string = re.sub(r"^\([a-z]\)", " ",string) 
		
		string = re.sub(r"\'s", r" \'s", string) 
		string = re.sub(r"\'ve", r" \'ve", string) 
		string = re.sub(r"n\'t", r" n\'t", string) 
		string = re.sub(r"\'re", r" \'re", string) 
		string = re.sub(r"\'d", r" \'d", string) 
		string = re.sub(r"\'ll", r" \'ll", string) 

		string = re.sub(r"\,[^\d\w]", " , ", string) 
		string = re.sub(r"\.[^\d\w]", " . ", string) 
		string = re.sub(r"\:", " : ", string) 
		string = re.sub(r"\;", " ; ", string) 
		string = re.sub(r"\!", " !", string) 
		string = re.sub(r"\(", " ( ", string) 
		string = re.sub(r"\)", " ) ", string) 
		string = re.sub(r"\? ", " ?", string) 
		# string = re.sub(r"--", " ", string) 
		# string = re.sub(r"-", " - ", string) 

		string = re.sub(r"\s{2,}", " ",string) 	
		string = string.rstrip().lstrip()
	return string



Bills = dict(bill_act = [], HR = [], date = [], people = [], short_title = [], summary = [], main = [])

with open(os.path.join(dir_raw_text, 'house_text_112a.txt')) as raw_bills:
	write = False
	write_date_people = False
	write_sum = False
	write_short = False
	date = ""
	people = ""
	main = ""
	summary = ""
	short_title = ""
	HR = ""
	bill_act = ""
	num_bill = 0
	
	for line in raw_bills:
	# for x in xrange(50000):
		# line = next(raw_bills).lstrip()
		line = line.lstrip()
		if line == "":
			pass


		if write_date_people:
			# Get Bill Date
			if re.match(r"^(January|February|March|April|May|June|July|August|September|October|November|December) \d+, \d{4}.*", line):
				date = re.match(r"^((January|February|March|April|May|June|July|August|September|October|November|December) \d+, \d{4}).*", line).group(1)
			else:
				people = people + line


		# Get Bill HR number
		if re.match(r'^H. ?R. ?(\d+)', line):
			HR = re.match(r'^H. ?R. ?(\d+)', line).group(1)
		# Get Bill summary
		if write_sum:
			summary = summary + line
			if re.match(r"\w", summary) and line == "":
				write_sum = False

		# Get Short title (continuing line)
		if write_short:
			if re.match(r"(.*)''.*", line):
				short_title = short_title + re.match(r"(.*)''.*", line).group(1) 
				write_short = False
			else:
				short_title = short_title + line

		# Get Short title (starting or only line)
		if short_title == "" and re.match(r".*This Act may be cited as the ``.*", line):
			if re.match(r".*This Act may be cited as the ``(.*)''.*", line):
				short_title = short_title + re.match(r".*This Act may be cited as the ``(.*)''", line).group(1)
			else:
				short_title = short_title + re.match(r".*This Act may be cited as the ``(.*)", line).group(1)
				write_short = True

		# Get main text
		if write:
			main = main + line

		# Detect symbol of the start of bio part
		if re.match(r"(IN THE HOUSE OF REPRESENTATIVES|IN THE SENATE OF THE UNITED STATES)", line):
			write_date_people = True

		# Detect symbol of the start of main text
		if (not write) and re.match("^(AN ACT|An Act|A BILL|A Bill|AMENDMENT|HOUSE AMENDMENT TO SENATE AMENDMENT)", line):
			if re.match("^(A BILL|A Bill)", line):
				bill_act = "Bill"
			if re.match("^(AN ACT|An Act)", line):
				bill_act = "Act"
			write_date_people = False
			write = True
			write_sum = True

		# Detect symbol of end of main text
		if re.match(r"^ENDENDEND", line):
			num_bill = num_bill + 1
			print "Number of Bills: ", num_bill

			Bills["main"].append(clean_string(main))
			Bills["summary"].append(clean_string(summary))
			Bills["short_title"].append(clean_string(short_title))
			Bills["date"].append(clean_string(date))
			Bills["people"].append(clean_string(people))
			Bills['HR'].append(HR)
			Bills['bill_act'].append(bill_act)
			write = False
			write_sum = False
			write_short = False
			HR = ""
			summary = ""
			main = ""
			short_title = ""
			date = ""
			people = ""
			bill_act = ""


import collections
counter = collections.Counter(Bills['HR'])


# Check duplicate
for key, item in counter.iteritems():
	if item > 1:
		print key, item
# Duplicates:
# 5652 2
# 4332 2
# 2647 2

with open("/Users/haohanchen/Dropbox/WORK/Research (now)/NN-vote/Data/house112.pickle", "wb") as out:
	pickle.dump(Bills, out)		
