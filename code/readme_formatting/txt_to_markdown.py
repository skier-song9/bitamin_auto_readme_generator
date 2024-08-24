# '''
# file_name = 'netflix'
# summ_dir = f'C:/Users/happy/Desktop/bitamin_auto_readme_generator/data/text_summarization/output/method3/{file_name}_text.txt'
# with open(summ_dir, 'r', encoding='utf-8') as file:
#     total_text = file.read()
# '''
import re

def txt_to_markdown(content):
    main_re = re.compile(r'<main>(.*?)</main>', re.DOTALL)
    sub_re = re.compile(r'<sub>(.*?)</sub>', re.DOTALL)
    content_re = re.compile(r'<content>(.*?)</content>', re.DOTALL)
    page_re = re.compile(r'<page>(.*?)</page>', re.DOTALL)

    # Find all occurrences of each tag
    subject = content.split("<subject>")[1].split("</subject>")[0].strip()
    team = content.split("<team>")[1].split("</team>")[0].strip().split(", ")
    index = content.split("<index>")[1].split("</index>")[0].strip().split(", ")
    mains = main_re.findall(content)

    # Convert to Markdown format
    readme_md = f"""# 🍊 {subject}
🗓️(프로젝트 진행기간을 입력하세요. ####.##.## ~ ####.##.##)
### 🚀 Team
{', '.join(team)}

## 🗒️Table of Contents
"""
    for i, section in enumerate(index, 1):
        readme_md += f"- [{section}](#section_{i})\n"
    readme_md += "<br>\n"
    for idx, main in enumerate(mains):
        main_text = main.strip()
        readme_md += f"<a name='section_{idx + 1}'></a>\n\n## 🔷 {main_text}\n\n"
        section_content = content.split(f"<main>{main_text}</main>")[1].split("<main>")[0]

        subs = sub_re.findall(section_content)
        contents = content_re.findall(section_content)
        pages = page_re.findall(section_content)

        # Add the subsections and content
        for sub, content_text, page in zip(subs, contents, pages):
            readme_md += f"#### {sub.strip()}\n\n"
            readme_md += f"- {content_text.strip()}\n\n"
            # readme_md += f"*Page: {page.strip()}*\n\n"
        readme_md += "<br>\n"

    return readme_md  # ,subject


# '''
# readme_md,subject = txt_to_markdown(total_text)
# # save readme file
# readme_dir = 'C:/Users/happy/Desktop/bitamin_auto_readme_generator/data/reademe_formatting'
# with open(f'{readme_dir}/{subject}.md', 'w', encoding='utf-8') as file:
#     file.write(readme_md)
#
# print("README.md file has been generated.")
# '''
