from ltp import LTP
import argparse

def parseargs():
	parser = argparse.ArgumentParser(description="evaluate the related scores")

	parser.add_argument("--ref", type=str, required=True,
						help="reference")
	parser.add_argument("--hypo", type=str, required=True,
						help="hypotheis")
	
	return parser.parse_args()

def srl_inference(hypo, ref):
	ltp = LTP()

	hypo_segment, hypo_hidden = ltp.seg([sen.split(" ") for sen in hypo], is_preseged=True)

	ref_segment, ref_hidden = ltp.seg([sen.split(" ") for sen in ref], is_preseged=True)

	hypo_srl = ltp.srl(hypo_hidden, keep_empty=False)

	ref_srl = ltp.srl(ref_hidden, keep_empty=False)

	return hypo_srl, ref_srl

def f_score(hypo_srl, ref_srl):

	predictions, reference, overlap = 0.0, 0.0, 0.0

	def get_dict(srl):
		dic_srl = {}
		for tri in srl:
			predicate = tri[0]
			argu = tri[1]
			dic_srl[predicate] = argu
		return dic_srl

	for hypo, ref in zip(hypo_srl, ref_srl):
		hypo_dic = get_dict(hypo)
		ref_dic = get_dict(ref)
		for key in hypo_dic:
			if key in ref_dic:
				common = len(set(hypo_dic[key]) & set(ref_dic[key]))
				overlap += common 
			predictions += len(hypo_dic[key])

		reference += sum(len(ref_dic[k]) for k in ref_dic)

	p = 100 * overlap / predictions if predictions > 0.0 else 0.0
	r = 100 * overlap / reference if reference > 0.0 else 0.0
	f_score = 2 * p * r / (p + r) if p + r > 0.0 else 0.0

	print("precision:", p)
	print("recall:", r)
	print("f score:", f_score)

	return f_score

def main(args):
	
	ref_sen=[]
	with open(args.ref,"r")as file:
		for line in file:
			ref_sen.append(line.strip('\n').lower())

	hypo_sen=[]
	with open(args.hypo, "r")as file:
		for line in file:
			hypo_sen.append(line.strip("\n").lower())

	hypo_srl, ref_srl = srl_inference(hypo_sen, ref_sen)

	f = f_score(hypo_srl, ref_srl)

if __name__ == '__main__':
    parsed_args = parseargs()
    main(parsed_args) 
