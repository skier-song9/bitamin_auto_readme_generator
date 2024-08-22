import numpy as np

def read_text_file_line_by_line(file_path):
    """
    Read the content of a text file line by line.
    
    :param file_path: Path to the text file
    :return: List of lines in the file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def edit_distance(reference, hypothesis):
    """
    Calculate the edit distance between two sequences.

    :param reference: The ground truth sequence
    :param hypothesis: The OCR system's output sequence
    :return: The edit distance value
    """
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=int)
    
    # Initialize the matrix
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j
    
    # Compute the edit distance matrix
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,    # Deletion
                          d[i][j - 1] + 1,    # Insertion
                          d[i - 1][j - 1] + cost)  # Substitution
    
    return d[len(reference)][len(hypothesis)]

def wer(reference, hypothesis):
    """
    Calculate the Word Error Rate (WER) between two texts.

    :param reference: The ground truth text
    :param hypothesis: The OCR system's output text
    :return: The WER value
    """
    reference = reference.split()
    hypothesis = hypothesis.split()
    
    edit_dist = edit_distance(reference, hypothesis)
    wer_value = edit_dist / float(len(reference))
    
    return wer_value

def cer(reference, hypothesis):
    """
    Calculate the Character Error Rate (CER) between two texts.

    :param reference: The ground truth text
    :param hypothesis: The OCR system's output text
    :return: The CER value
    """
    reference = list(reference.replace(" ", ""))
    hypothesis = list(hypothesis.replace(" ", ""))
    
    edit_dist = edit_distance(reference, hypothesis)
    cer_value = edit_dist / float(len(reference))
    
    return cer_value

# Example usage
reference_path = 'C:/Users/happy/Desktop/bitamin_auto_readme_generator/data/object_detection/output/netflix-stock/GT-netflix-stock.txt'
hypothesis_path = 'C:/Users/happy/Desktop/bitamin_auto_readme_generator/data/object_detection/output/netflix-stock/ocr_result/paddle-ocr_netflix-stock_text.txt'

# Read texts from files
reference_lines = read_text_file_line_by_line(reference_path)
hypothesis_lines = read_text_file_line_by_line(hypothesis_path)

# Ensure both files have the same number of lines
assert len(reference_lines) == len(hypothesis_lines), "Files must have the same number of lines"

# Calculate WER and CER for each line and average them
total_wer = 0
total_cer = 0
for ref_line, hyp_line in zip(reference_lines, hypothesis_lines):
    ref_line = ref_line.strip()
    hyp_line = hyp_line.strip()
    total_wer += wer(ref_line, hyp_line)
    total_cer += cer(ref_line, hyp_line)

average_wer = total_wer / len(reference_lines)
average_cer = total_cer / len(reference_lines)
print(f'Average WER: {average_wer:.4f}')
print(f'Average CER: {average_cer:.4f}')
