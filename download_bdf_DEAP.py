import os
import subprocess
import logging
import json

logging.basicConfig(level=logging.INFO)

class downloader_and_unzipper:
	def __init__(self):
		with open('login.json', 'r') as JSON:
			json_dict = json.load(JSON)

		self.username = json_dict['username']
		self.password = json_dict['password']
		self.links = self.gen_links()

	def gen_links(self):
		link_dict = {}
		# 60 zip files
		for a_number in range(1, 61):

			number_str = str(a_number)
			zero_filled_number = number_str.zfill(3)
			filename = 'data_original.zip.{}'.format(zero_filled_number)
			filepath = os.path.join(os.getcwd(), 'zip_files', filename)
			# link example: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/data_original.zip.001
			wget_cmd = 'wget https://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/{} --user {} --password={} -O {}'.format(filename, self.username, self.password, filepath)
			link_dict[wget_cmd] = filename
		return link_dict

	def download_and_process(self):

		zip_folder = os.path.join(os.getcwd(), 'zip_files')
		if not os.path.exists(zip_folder):
			os.makedirs(zip_folder, exist_ok=True)

		for link in self.links.keys():
			filename = self.links[link]

			if not os.path.isfile(filename):
				logging.info("downloading file: " + filename)
				subprocess.call(link, shell=True)

		merged_zip_filename = 'data_original_full.zip'
		join_zips_cmd = 'cat {} > {}'.format( os.path.join(os.getcwd(), 'zip_files', 'data_original.z*'), merged_zip_filename )
		zip_cmd = 'zip -f {}'.format(merged_zip_filename)
		unzip_cmd = 'unzip {}'.format(merged_zip_filename)
		os.system(join_zips_cmd)
		os.system(zip_cmd)
		os.system(unzip_cmd)
		os.remove(merged_zip_filename)
		# os.remove(bdf_folder)

if __name__ == "__main__":
	downloader = downloader_and_unzipper()
	downloader.download_and_process()