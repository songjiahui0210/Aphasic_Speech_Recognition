import zipfile

def open_zip_file(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall()

open_zip_file('../../../data/aphasia/English/Aphasia/Kempler/Kempler.zip')
# open_zip_file('../../../data/aphasia/English/Control/Kempler/Kempler.zip')