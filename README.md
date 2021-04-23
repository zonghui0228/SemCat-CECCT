# SemCat-CECCT
> Semantic categorization of Chinese eligibility criteria in clinical trials using machine learning methods.

## data collection

We downloaded the clinical trials registration files from website of Chinese Clinical Trials Registry ([ChiCTR](http://www.chictr.org.cn/)), and extracted the eligibility criteria text, including Chinese inclusion criteria(CIC), English inclusion criteria(EIC), Chinese exclusion criteria(CEC) and English exclusion criteria(EEC). An eligibility criteria example (registration number: [ChiCTR1800016069](http://www.chictr.org.cn/showproj.aspx?proj=27068)) shows below:

| **Chinese inclusion criteria**                               | **English inclusion criteria**                               |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 1）首次于本中心行肝癌切除术，术后组织病理学证实HCC；<br/>2）术前血清HBsAg（+）；<br/>3）血清HBV-DNA低于检测下限（内标法）；<br/>4）肿瘤未侵犯门静脉、肝静脉或胆管的主要分支；<br/>5）肝功能Child-Pugh A或B级；<br/>6）年龄18～65岁，性别不限；<br/>7）单发肿瘤者，最大径≤5cm；多发肿瘤者，瘤体数≤3个且各瘤体最大径≤3cm； | 1. patient receives hepatectomy of HCC, which was confirmed with pathology;<br/>2. serum HBsAg（+）is confirmed preoperatively;<br/>3. serum HBV-DNA is lower than minimum of detection;<br/>4. tumor did not invade potal vein,hepatic vein or major branch of biliary tract;<br/>5. liver function with Child-Pugh A or B;<br/>6. aged from 18-65 years male or female;<br/>7. single tumor≤5cm.for multiple tumors,number of tumor≤3 and every tumor≤3cm. |
| **Chinese exclusion criteria**                               | **English exclusion criteria**                               |
| 1）肝细胞癌合并肝外转移；<br/>2) 近1年内有抗病毒治疗；<br/>3）合并其他部位恶性肿瘤，或其他器官功能衰竭；<br/>4）合并其它肝炎病毒重叠感染；<br/>5）术前接受TACE或其他抗肿瘤治疗；<br/>6）肝功能Child-Pugh C级<br/>7）存在顽固性腹水或肝性脑病；<br/>8）因精神病、心理性疾病或其他原因不能配合治疗或不能完成疗程者。 | 1. HCC with metastasis out of the liver;<br/>2. antiviral therapy to HBV within one year;<br/>3. with other malignancy or organ failure;<br/>4. combined with other hepatic virus infection;<br/>5. recieved TACE or other antitumor therapy;<br/>6. liver function with Child-Pugh C;<br/>7. refractory Ascites or hepatic encephalopathy;<br/>8. patient who can not cooperate due to mental diseases or other reasons. |

The eligibility criteria text is organized with multiple sentences. After sentence segmentation and filter, we randomly selected 19185 sentences used in the unsupervised hierarchical clustering section for criteria categories induction, and 38341 sentences used in supervised criteria classification section for the classification capacity validation of induced semantic categories.

## clustering
We used the agglomerative hierarchical clustering algorithm, works in a “bottom-up” manner, to cluster constructed semantic feature matrix and generated clusters based on criteria sentences similarity..

1. get 19185 Chinese-English eligibility criteria sentences pairs.
2. get UMLS semantic types of English eligibility criteria sentences using MetaMap.
3. convert criteria sentences to feature matrix based on UMLS semantic types.
4. perform hierarchical clustering and summarize semantic categories.

## classification

To assess the classification capacity of our induced semantic categories used for the Chinese eligibility criteria sentences classification, we randomly selected 38341 Chinese eligibility criteria sentences and implemented 9 classification algorithms.

- Machine learning algorithms
  - NB
  - kNN
  - LR
  - SVM
- Deep learning algorithms
  - CNN
  - RNN
  - FastText
- Pre-trained language models
  - BERT
  - ERNIE

classification results:

<table>
   <tr>
      <td colspan="2" rowspan="2">Models</td>
      <td colspan="3">Macro-average</td>
      <td colspan="3">Micro-average</td>
   </tr>
   <tr>
      <td>precision</td><td>recall</td><td>F1-score</td><td>precision</td><td>recall</td>
      <td>F1-score</td>
   </tr>
   <tr>
      <td rowspan="4">Machine learning algorithms</td>
      <td>NB</td>
      <td>0.5398</td><td>0.7403</td><td>0.5965</td>
      <td>0.6312</td><td>0.6312</td><td>0.6312</td>
   </tr>
   <tr>
      <td>kNN</td><td>0.7531</td><td>0.6693</td><td>0.6948</td>
      <td>0.7632</td><td>0.7632</td><td>0.7632</td>
   </tr>
   <tr>
      <td>LR</td>
      <td>0.8017</td><td>0.7574</td><td>0.7732</td>
      <td>0.8173</td><td>0.8173</td><td>0.8173</td>
   </tr>
   <tr>
      <td>SVM</td>
       <td><b>0.8196</b></td><td>0.7712</td><td>0.7899</td>
      <td>0.8293</td><td>0.8293</td><td>0.8293</td>
   </tr>
   <tr>
      <td rowspan="3">Deep learning algorithms</td>
      <td>CNN</td>
      <td>0.8004</td><td>0.6951</td><td>0.7258</td>
      <td>0.8142</td><td>0.8142</td><td>0.8142</td>
   </tr>
   <tr>
      <td>RNN</td>
      <td>0.7837</td><td>0.6925</td><td>0.7170</td>
      <td>0.8138</td><td>0.8138</td><td>0.8138</td>
   </tr>
   <tr>
      <td>FastText</td>
      <td>0.7645</td><td>0.7188</td><td>0.7341</td>
      <td>0.8182</td><td>0.8182</td><td>0.8182</td>
   </tr>
   <tr>
      <td rowspan="2">Pre-trained language models</td>
      <td>BERT</td>
      <td>0.7994</td><td>0.8023</td><td>0.7958</td>
      <td>0.8447</td><td>0.8447</td><td>0.8447</td>
   </tr>
   <tr>
      <td>ERNIE</td>
       <td>0.7964</td><td><b>0.8074</b></td><td><b>0.7980</b></td>
       <td><b>0.8484</b></td><td><b>0.8484</b></td><td><b>0.8484</b></td>
   </tr>
</table>

we also organized a shared task on [fifth China Conference on Health Information Processing (CHIP 2019)](https://github.com/zonghui0228/chip2019task3). As organizers, we released our labeled data and defined 44 categories. A total 75 teams participated in the task and 27 of them submitted results. The best performing system achieved a macro F1 score of **0.81** by applied multiple pre-trained language models and ensemble modeling.

## How to cite

Zong, H., Yang, J., Zhang, Z. *et al.* Semantic categorization of Chinese eligibility criteria in clinical trials using machine learning methods. *BMC Med Inform Decis Mak* **21,** 128 (2021). https://doi.org/10.1186/s12911-021-01487-w

## Contacts

Zong Hui , [zonghui@tongji.edu.cn](mailto:zonghui@tongji.edu.cn)

Tongji University, Shanghai, 200092, China

