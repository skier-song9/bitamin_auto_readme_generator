from text_summ_utils import *
import argparse
import stat

if __name__ == '__main__':
    # 매개변수 받기
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc') # 원본 문서가 있는 디렉토리 전달
    parser.add_argument('--summ') # 요약 텍스트가 있는 디렉토리 전달하기
    # 인자 파싱
    args = parser.parse_args()

    gpt = GPT(api_filepath = OPENAI_API_FILEPATH)
    doc_dir = os.path.join(root_absdir, 'data','object_detection','output',args.doc)
    doc_files = os.listdir(doc_dir)
    summ_dir = os.path.join(root_absdir, 'data','text_summarization','output',args.summ)

    scores_df = pd.DataFrame(columns = ['pdf','relevance','coherence','consistency','fluency','summ/doc_ratio'])

    for filename in doc_files:
        print(filename,"started.")
        with open(os.path.join(doc_dir, filename), 'r', encoding='utf-8') as f:
            document = f.readlines()
        if os.path.exists(os.path.join(summ_dir, filename)):
            with open(os.path.join(summ_dir, filename), 'r', encoding = 'utf-8') as f:
                summary = f.readlines()
        else:
            print("⚠️Summary doesn't exist")
            continue
        # preprocess
        document = ''.join(document)
        document = erase_tag(document, 'p.\d*')
        # document = document.replace('\n','')
        summary = ''.join(summary)
        # for tag in ['subject','team','index']:
        #     summary = remove_xml_tags(summary, tag)
        # summary = erase_tag(summary,'[^>]+')  
        summary = remove_xml_tags(summary, 'page')

        # Evaluate
        rel, coh, cons, flu =gpt.get_geval_score(document, summary, model='gpt-4o-mini')
    
        new_row = pd.DataFrame(data=[[filename.replace('.txt',''),rel,coh,cons,flu,round(len(summary)/len(document)*100,2)]],
                              columns=scores_df.columns)
        scores_df = pd.concat([scores_df, new_row], axis=0)

    # Save file
    t = time.localtime()
    savepath = os.path.join(root_absdir,'data','text_summarization','output','g-evals',f'g-eval_{args.summ}_{t.tm_hour}{t.tm_min}.csv')
    scores_df.to_csv(savepath, index=False)
    