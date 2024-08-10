import openai
import json
import requests
from openai import OpenAI

import pandas as pd
import glob,os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import re
import argparse
import stat
import math
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering,KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, CanineTokenizer, CanineModel

import warnings
warnings.filterwarnings("ignore")

ROOT = 'bitamin_auto_readme_generator'
RANDOM_STATE = 142
PAGE_PATTERN = '<p.\d*>'
OPENAI_API_FILEPATH = "../assets/openai_api_key.json"

def split_by_pages(filepath, encoding='utf-8'):
    """
    filepath : ocr 결과 txt 파일 경로 전달

    return : page별로 구분된 하나의 리스트를 반환
    """
    with open(filepath,'r',encoding=encoding) as f:
        data = f.readlines()
    text = []
    page_text = []
    for d in data:
        if re.compile(PAGE_PATTERN).match(d):
            if len(page_text)>0:
                text.append(' '.join(page_text))
            page_text = []
        page_text.append(d)
    text.append(''.join(page_text))
    return text

def erase_tag(text, tag):
    """
    text : split_by_page로 얻은 텍스트 리스트 또는 텍스트
    tag : 지우고 싶은 <tag>
    """
    tag_pattern = re.compile(f'<{tag}>|</{tag}>')
    if isinstance(text, list):
        text_ = [tag_pattern.sub('',x) for x in text]
        return text_
    else:
        text_ = tag_pattern.sub('',text)
        return text_

def extract_text_between_tag(text, tag):
    """
    text : split_by_page로 얻은 텍스트 리스트
    tag : <tag> 사이의 텍스트를 추출
    """
    # Create a regex pattern for the specified tag
    pattern = f'<{tag}>(.*?)</{tag}>'
    # Use re.findall to extract all occurrences between the specified tags
    matches = re.findall(pattern, text, re.DOTALL)
    return matches
    
def extract_index_page(textlist, table_of_contents, threshold = 0.35):
    '''
    textlist : 첫 5개 문장만 전달한다.
    table_of_contents : 추출한 목차를 전달

    return : textlist에서 목차에 해당하는 문장의 index를 반환
    '''
    toc = ', '.join(table_of_contents)
    results = np.array([])
    for sentence in erase_tag(textlist,"p.\d*"):
        sentence = sentence.replace('\n','')
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([toc,sentence])
        cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])
        results = np.append(results, cosine_sim)
    
    max_index = results.argmax()
    max_cs = results[max_index]
    if max_cs <= threshold:
        return results, 0
    return results, max_index

class GPT():
    __classname__ = "OpenAI"
    api_key = ''
    client = None    
    def __init__(self, api_filepath):
        with open(api_filepath,'r') as f:
            ak = json.load(f)
        self.api_key = ak['OPENAI_API_KEY']
        self.client = OpenAI(api_key=self.api_key)
        
    def get_chat_completion(self, msg, model='gpt-4o-mini', temperature = 0):
        response = self.client.chat.completions.create(
            model = model,
            messages = msg,
            temperature = temperature
        )
        return response.choices[0].message.content

    def get_embedding(self, sentence, model="text-embedding-3-small"):
       '''
       - pricing : text-embedding-3-small = $0.02/1M tokens
           텍스트가 많은 pdf는 대략 5,000 tokens -> pdf 200개에 0.02 달러(25~30원).
       text : 한 문장
       return : 한 문장에 대한 embedding (output dimension = 1536)
       '''
       return self.client.embeddings.create(input = sentence, model=model).data[0].embedding


class TextSummerizer():
    model_dict = dict()
    def __init__(self, gpt_client):
        ### Initiate Embedding Models
        # 하나의 sentence를 통째로 embedding하는 모델들
        se_model = SentenceTransformer('sentence-transformers/LaBSE') # BERT 기반 문장 임베딩 모델
        se_model.__classname__ = "LaBSE"
        gist_model = SentenceTransformer("avsolatorio/GIST-Embedding-v0", 
                                    revision=None)
        gist_model.__classname__ = "GIST-Embedding-v0" 
        self.gpt = gpt_client
        # tokenizer로 토큰화 후 embedding하는 모델들
        convbert_tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base")
        convbert_model = AutoModel.from_pretrained("YituTech/conv-bert-base")
        convbert_model.__classname__ = "ConvBERT" 
        canine_tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
        canine_model = CanineModel.from_pretrained('google/canine-c')
        canine_model.__classname__ = "Canine-C"
        self.model_dict = {
            'sentence':[se_model, gist_model, self.gpt],
            'token':[(convbert_tokenizer,convbert_model),(canine_tokenizer,canine_model)]
        }

    def pca_best_component(self, x):
        '''
        최적의 PCA component값을 찾는 함수
        '''
        pca_optimize = PCA()
        pca_optimize.fit(x)
        # 누적 설명된 분산 비율 계산
        cumulative_variance = np.cumsum(pca_optimize.explained_variance_ratio_)
        # 99% 이상 설명력을 갖는 주성분 개수 계산
        n_components = np.argmax(cumulative_variance >= 0.99) + 1
        return n_components

    def find_optimal_k(self, max_cluster, x):     
        '''
        K-means, Spectral clustering에서 최적의 K값을 찾는 함수
        '''
        # cluster의 최솟값은 3으로 고정하고 clusterlist를 만든다.
        min_cluster = 4
        cluster_lists = [x for x in range(min_cluster,max_cluster)]
        
        # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
        n_cols = len(cluster_lists)
        
        results = []
        # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
        for ind, n_cluster in enumerate(cluster_lists):
            
            # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
            clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
            cluster_labels = clusterer.fit_predict(x)
            
            sil_avg = silhouette_score(x, cluster_labels)
            sil_values = silhouette_samples(x, cluster_labels)
            results.append([sil_avg, np.var([x if x>=0 else 0 for x in sil_values])])
    
        ### 결과를 토대로 최적의 K 값을 도출
        # 1. 실루엣 계수의 평균과 분산에 대해 MinMax 정규화
        results_df = pd.DataFrame(data=results,columns=['avg','var']) 
        mms = MinMaxScaler()
        results_df[['avg_norm', 'var_norm']] = mms.fit_transform(results_df[['avg', 'var']])
        # 2. 표준화된 var을 1에서 빼서 "분산은 작은 값이 좋음"을 반영
        results_df['var_norm'] = 1 - results_df['var_norm']
        # 3. 가중 평균 계산 (평균에 대한 가중치: 0.65, 분산에 대한 가중치: 0.35 -> 가중치에 대한 근거는 경험적 판단.)
        results_df['score'] = results_df['avg_norm'] * 0.65 + results_df['var_norm'] * 0.35    
        sorted_results = results_df.sort_values('score',ascending=False)
        optimal_k = sorted_results.index[0]+min_cluster
        return optimal_k

    def clustering(self, text, n_clusters, random_state=RANDOM_STATE):
        '''
        text : pdf 문서를 page별로 split하고, <p>태그와 \n 문자열을 없앤 텍스트 리스트 전달.(=textlist)
        n_clusters : 클러스터의 개수를 지정 -> extract_text_between_tag(text,'index')로 추출한 인덱스의 개수 + 2
    
        return
            - total_metrics : 모델별, 클러스터링 알고리즘별 클러스터링 결과를 SS, CHI로 평가지표로 측정한 결과에 대한 데이터프레임
            - total_dict : model_name을 key값으로 text, cluster label, embedding 컬럼으로 하는 데이터프레임을 value로 갖는 딕셔너리
        '''
        total_metrics = pd.DataFrame(columns=['model','kmeans_ss','spectral_ss','kmeans_chi','spectral_chi'])
        total_dict = {}
        max_cluster = len(text)-1
        # sentence 단위로 임베딩하는 모델의 경우
        print("--Sentence Embedding Methods--")
        for model in self.model_dict['sentence']:
            model_name = model.__classname__
            if model_name == 'Encoding':
                text_embeddings = [model.encode(t) for t in text]
            elif model_name in set(['LaBSE','GIST-Embedding-v0']):
                text_embeddings = model.encode(text, convert_to_tensor=True)
                text_embeddings = text_embeddings.cpu().detach()
            elif model_name == 'OpenAI':
                text_embeddings = []
                for t in text:
                    embedding = model.get_embedding(t)
                    text_embeddings.append(embedding)
                text_embeddings = np.array(text_embeddings)
            else:
                text_embeddings = model.encode(text)
            ### 차원 축소 : PCA + t-SNE
            # PCA n_component 구하기
            optimal_component = self.pca_best_component(text_embeddings)
            # PCA
            text_embeddings = PCA(n_components=optimal_component, random_state=random_state).fit_transform(text_embeddings)
            # t-SNE (3차원으로 축소하는 것으로 고정)
            # perplexity는 일반적으로 데이터 개수의 3분의 1 이하
            general_perplexity = int(text_embeddings.shape[0]/3)
            # 최솟값은 5, 최댓값은 50으로 제한
            if general_perplexity<5:
                general_perplexity = 5
            elif general_perplexity>50:
                general_perplexity=50
            text_embeddings = TSNE(n_components=3, perplexity=general_perplexity,
                                   random_state=random_state).fit_transform(text_embeddings)
            # 최적의 k값 찾기 [x] >> pdf의 인덱스 개수에 맞춰 n_clusters를 설정
            # optimal_k = find_optimal_k(max_cluster=max_cluster,x=text_embeddings)
            # kmeans와 spectral clustering 진행
            kmeans = KMeans(n_clusters=n_clusters, 
                    init='k-means++', # centroid들을 서로 최대한 멀리 배치하는 initialisation 방식
                    max_iter = 500,
                    random_state = random_state)
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                                 random_state=random_state)
            kmeans.fit(text_embeddings)
            spectral.fit(text_embeddings)
            # 평가지표 계산
            new_row = pd.DataFrame(data=[[
                model_name, # model name
                silhouette_score(text_embeddings, kmeans.labels_), # kmeans ss
                silhouette_score(text_embeddings, spectral.labels_), # spectral ss
                calinski_harabasz_score(text_embeddings, kmeans.labels_), # kmeans chi
                calinski_harabasz_score(text_embeddings, spectral.labels_) # spectral chi
            ]], columns = total_metrics.columns)
    
            # Save results
            total_metrics = pd.concat([total_metrics, new_row],axis=0,ignore_index=True)
            temp_df = pd.DataFrame(data=[
                text, kmeans.labels_, spectral.labels_
            ]).transpose()
            temp_df.columns = ['text','kmeans','spectral']
            total_dict[model_name] = pd.concat([
                temp_df,pd.DataFrame(text_embeddings)
            ],axis=1)
    
            # garbage collection > 별 의미 없는 듯
            # gc.collect()
    
        # token 단위로 임베딩하는 모델의 경우
        print("--Token Embedding Methods--")
        for tokenizer, model in self.model_dict['token']:
            model_name = model.__classname__
            if model_name == 'Canine-C':
                token = tokenizer(erase_tag(text,'p.\d*'), padding='longest', truncation=True, return_tensors='pt')
            else: # ConvBERT
                token = tokenizer(erase_tag(text,'p.\d*'), padding='longest',return_tensors='pt')
            text_embeddings = model(**token).last_hidden_state.detach()
            flattened = text_embeddings.view(text_embeddings.shape[0],-1)
            ### 차원 축소 : PCA + t-SNE
            # PCA n_component 구하기
            optimal_component = self.pca_best_component(flattened)
            # PCA
            text_embeddings = PCA(n_components=optimal_component, random_state=random_state).fit_transform(flattened)
            # t-SNE (3차원으로 축소하는 것으로 고정)
            # perplexity는 일반적으로 데이터 개수의 3분의 1 이하
            general_perplexity = int(text_embeddings.shape[0]/3)
            # 최솟값은 5, 최댓값은 50으로 제한
            if general_perplexity<5:
                general_perplexity = 5
            elif general_perplexity>50:
                general_perplexity=50
            text_embeddings = TSNE(n_components=3, perplexity=general_perplexity,
                                   random_state=random_state).fit_transform(text_embeddings)
    
            # 최적의 k값 찾기 [x] >> pdf의 인덱스 개수에 맞춰 n_clusters를 설정
            # optimal_k = find_optimal_k(max_cluster=max_cluster,x=text_embeddings)
            # kmeans와 spectral clustering 진행
            kmeans = KMeans(n_clusters=n_clusters, 
                    init='k-means++', # centroid들을 서로 최대한 멀리 배치하는 initialisation 방식
                    max_iter = 500,
                    random_state = random_state)
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                                 random_state=random_state)
            kmeans.fit(text_embeddings)
            spectral.fit(text_embeddings)
            # 평가지표 계산
            new_row = pd.DataFrame(data=[[
                model_name, # model name
                silhouette_score(text_embeddings, kmeans.labels_), # kmeans ss
                silhouette_score(text_embeddings, spectral.labels_), # spectral ss
                calinski_harabasz_score(text_embeddings, kmeans.labels_), # kmeans chi
                calinski_harabasz_score(text_embeddings, spectral.labels_) # spectral chi
            ]], columns = total_metrics.columns)
    
            # Save results
            total_metrics = pd.concat([total_metrics, new_row],axis=0,ignore_index=True)
            temp_df = pd.DataFrame(data=[
                text, kmeans.labels_, spectral.labels_
            ]).transpose()
            temp_df.columns = ['text','kmeans','spectral']
            total_dict[model_name] = pd.concat([
                temp_df,pd.DataFrame(text_embeddings)
            ],axis=1)
    
            # garbage collection
            # gc.collect()
    
        return total_metrics, total_dict

    def best_model_n_algo(self, metrics_df):
        """
        metrics_df : clustering함수 결과로 얻은 total_metrics
    
        return : total_metrics에서 가장 성능이 좋은 model과 클러스터링 알고리즘을 반환 
        -> total_dict에서 해당 모델 이름과 클러스터링 알고리즘 이름으로 클러스터 결과를 찾을 수 있음.
        """
        metrics_df['kmeans']=metrics_df['kmeans_ss']+metrics_df['kmeans_chi']
        metrics_df['spectral']=metrics_df['spectral_ss']+metrics_df['spectral_chi']
        
        max_value = metrics_df[['spectral','kmeans']].max().max()
        # 최대값을 가진 행과 열 찾기
        max_value_row_col = metrics_df[['spectral', 'kmeans']].apply(lambda x: x == max_value).stack()
        max_value_row_col = max_value_row_col[max_value_row_col].index[0]
        model_name = metrics_df.loc[max_value_row_col[0],'model']
        clustering_algo = max_value_row_col[1]
        return model_name, clustering_algo

    def renew_cluster_byorder(self, total_dict, best):
        '''
        total_dict : clustering 함수 반환값인 total_dict
        best = best_model_n_algo 함수 반환값인 (best_model, best_algo) 튜플
            - best_model : best_model_n_algo 결과로 얻은 model_name
            - best_algo : best_model_n_algo 결과로 얻은 clustering_algo
    
        return df : 순서가 엉켜있는 cluster label을 재정렬한 결과를 반환
        '''
        best_model, best_algo = best
        df = total_dict[best_model][['text',best_algo]]
        cnum = len(df[best_algo].unique())
        checklist = []
        for c in df[best_algo]:
            if c not in checklist:
                checklist.append(c)
            if len(checklist) == cnum:
                break
        renew_c = list(range(cnum))
        df['new_cluster'] = df[best_algo].apply(lambda x:renew_c[checklist.index(x)])
        return df


if __name__ == '__main__':
    # 매개변수 받기
    parser = argparse.ArgumentParser()
    parser.add_argument('--input') # ocr_samples_txt 전달하기
    parser.add_argument('--output') # cluster_n_summary 전달하기
    parser.add_argument('--temperature')
    # 인자 파싱
    args = parser.parse_args()

    root_absdir = os.getcwd().split(ROOT)[0]+ROOT
    input_dir = os.path.join(root_absdir,'data','object_detection','output',args.input)
    output_dir = os.path.join(root_absdir,'data','text_summarization','output',args.output)

    # start logic
    gpt = GPT(api_filepath=OPENAI_API_FILEPATH)
    text_summarization = TextSummerizer(gpt_client=gpt)
    
    for file in os.listdir(input_dir):
        print(file,"work started.")
        start_time = time.time()
        filepath = os.path.join(input_dir, file)
        # load text
        textlist = split_by_pages(filepath=filepath, encoding='utf-8')
        # 첫 5페이지, 마지막 3페이지로부터 "주제, 팀원, 목차" 추출하기
        msg = [
            {"role": "system", 
             "content": """Refer to the document provided between the triple quotes. This document is a text conversion of a PDF file and is structured into passages separated by '\n'. Each passage corresponds to a different textbox in the PDF. Note that repeated passages are considered redundant and should be ignored in your response.

Perform the following tasks:
1. **Identify the main subject of the document**: Select the passage that best represents the overall theme or topic of the document. If multiple passages are selected, combine them together.
2. **Extract names of individuals**: Identify and list all names mentioned in the document, separated by commas.
3. **Extract the table of contents (index)**: Identify and list the contents or sections of the document, separated by commas.

Respond in the following format:
<subject>Result of Task 1</subject>
<team>Result of Task 2</team>
<index>Result of Task 3</index>

If any result is unclear or not found, indicate it as None.
             """
            },
            {"role": "user", 
             "content": f'"""{" ".join(textlist[:5]) + " ".join(textlist[-3:])}"""'
            }
        ]
        sub_team_index = gpt.get_chat_completion(msg, model='gpt-4o-mini')
        # 목차 개수 추출 -> 목차 개수를 cluster 개수로 설정
        table_of_contents = extract_text_between_tag(sub_team_index,'index')[0].split(',')
        nclusters = len(table_of_contents) 
        # 목차 이후의 페이지만 클러스터링하기
        _, index_page = extract_index_page(textlist[:5], table_of_contents, threshold=0.35)
        textlist = textlist[index_page+1:] # 목차 페이지 이후부터 클러스터링
        # 텍스트에서 page tag와 \n 문자열을 제거
        textlist_preprocessed = [text.replace('\n',' ') for text in erase_tag(textlist,'p.\d*')]
        # clustering
        print("Clustering Started.")
        c_metrics, c_dfs = text_summarization.clustering(text=textlist_preprocessed,
                                     n_clusters=nclusters, random_state=RANDOM_STATE)
        print("✅Clustering Finished.")
        # 성능이 제일 좋은 임베딩 모델과 클러스터링 알고리즘 선택
        best_model, best_algo = text_summarization.best_model_n_algo(c_metrics)
        # del c_metrics # 더이상 불필요한 변수
        # cluster label 재정렬
        c_renew = text_summarization.renew_cluster_byorder(total_dict=c_dfs,
                                                           best=(best_model,best_algo))
        # del c_dfs # 더이상 불필요한 변수

        # 클러스터별로 텍스트 묶기
        c_dict = dict(zip(list(range(nclusters)),['' for _ in range(nclusters)]))
        for idx in c_renew.index:
            new_cluster = c_renew.iloc[idx]['new_cluster']
            c_dict[new_cluster] += c_renew.iloc[idx]['text']

        # 클러스터별로 텍스트 요약
        i = 0
        summarizations = []
        for key in range(nclusters): # 목차에 해당하는 cluster만 요약함.
            table_of_content = table_of_contents[i]
            i += 1
            ctext = c_dict[key]
            msg = [
                {"role": "system", 
                 "content": f"""Your task is to 'Summarize' the given text which is in triple quote with a focus on the key points related to '{table_of_content}'. Extract and summarize the content for each relevant subtitle. Answer in Korean and follow the format provided below.

<main>{table_of_content}</main>
<subtitle>[Extracted Subtitle]</subtitle>
<content>Summary of the content under this subtitle</content>

Instructions:
1. Identify and extract subtitles from the text that relate to '{table_of_content}'.
2. Summarize the content under each subtitle within one or two sentences, focusing on the most important details and main ideas.
3. Provide your summary in Korean, maintaining clarity and coherence.
4. Use the specified format for organizing your response.
                """
                },
                {"role": "user", 
                 "content": f'"""{ctext}"""'
                }
            ]
            summarizations.append(gpt.get_chat_completion(msg, model='gpt-4o-mini', temperature=int(args.temperature)))
        
        output_filepath = os.path.join(output_dir, file)
        with open(output_filepath,'w', encoding='utf-8') as f:
            f.write(sub_team_index)
            for s in summarizations:
                f.write('\n')
                f.write(s)
        print(f"📢 {file} finished in {time.time()-start_time:0.1f} seconds")
    # end for
    

    
    
    
    