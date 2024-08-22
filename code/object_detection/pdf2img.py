import os
from pdf2jpg import pdf2jpg

ROOT = 'bitamin_auto_readme_generator'

def convert_and_rename_pdf(pdf_name):
    root_absdir = os.getcwd().split(ROOT)[0]+ROOT
    input_dir = os.path.join(root_absdir,'data','object_detection','input',f'{pdf_name}.pdf')
    output_dir = os.path.join(root_absdir,'data','object_detection','input')
    new_output_dir = os.path.join(output_dir, pdf_name)

    pdf2jpg.convert_pdf2jpg(input_dir, output_dir, pages="ALL")
    generated_dir = os.path.join(output_dir, pdf_name + '.pdf_dir')

    if os.path.exists(generated_dir):
        os.rename(generated_dir, new_output_dir)
        for idx, filename in enumerate(sorted(os.listdir(new_output_dir))):
            old_file = os.path.join(new_output_dir, filename)
            new_filename = f"{idx:02}.jpg"
            new_file = os.path.join(new_output_dir, new_filename)
            os.rename(old_file, new_file)
        print(f"Conversion completed.")
    else:
        print(f"Error: Generated directory '{generated_dir}' not found.")

'''
example >
convert_and_rename_pdf('variation')
'''

