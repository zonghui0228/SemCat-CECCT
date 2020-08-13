# SemCat-CECCT
> Semantic categorization of Chinese eligibility criteria in clinical trials using machine learning methods.

## data collection

We downloaded the clinical trials registration files from website of Chinese Clinical Trials Registry ([ChiCTR](http://www.chictr.org.cn/)), and extracted the eligibility criteria text, including Chinese inclusion criteria(CIC), English inclusion criteria(EIC), Chinese exclusion criteria(CEC) and English exclusion criteria(EEC). An eligibility criteria example (registration number: [ChiCTR1800016069](http://www.chictr.org.cn/showproj.aspx?proj=27068)) shows below:

| **Chinese inclusion criteria**                               | **English inclusion criteria**                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1）首次于本中心行肝癌切除术，术后组织病理学证实HCC；<br/>2）术前血清HBsAg（+）；<br/>3）血清HBV-DNA低于检测下限（内标法）；<br/>4）肿瘤未侵犯门静脉、肝静脉或胆管的主要分支；<br/>5）肝功能Child-Pugh A或B级；<br/>6）年龄18～65岁，性别不限；<br/>7）单发肿瘤者，最大径≤5cm；多发肿瘤者，瘤体数≤3个且各瘤体最大径≤3cm； | 1. patient receives hepatectomy of HCC, which was confirmed with pathology;<br/>2. serum HBsAg（+）is confirmed preoperatively;<br/>3. serum HBV-DNA is lower than minimum of detection;<br/>4. tumor did not invade potal vein,hepatic vein or major branch of biliary tract;<br/>5. liver function with Child-Pugh A or B;<br/>6. aged from 18-65 years male or female;<br/>7. single tumor≤5cm.for multiple tumors,number of tumor≤3 and every tumor≤3cm. |
| **Chinese exclusion criteria**                               | **English exclusion criteria**                               |
| 1）肝细胞癌合并肝外转移；<br/>2) 近1年内有抗病毒治疗；<br/>3）合并其他部位恶性肿瘤，或其他器官功能衰竭；<br/>4）合并其它肝炎病毒重叠感染；<br/>5）术前接受TACE或其他抗肿瘤治疗；<br/>6）肝功能Child-Pugh C级<br/>7）存在顽固性腹水或肝性脑病；<br/>8）因精神病、心理性疾病或其他原因不能配合治疗或不能完成疗程者。 | 1. HCC with metastasis out of the liver;<br/>2. antiviral therapy to HBV within one year;<br/>3. with other malignancy or organ failure;<br/>4. combined with other hepatic virus infection;<br/>5. recieved TACE or other antitumor therapy;<br/>6. liver function with Child-Pugh C;<br/>7. refractory Ascites or hepatic encephalopathy;<br/>8. patient who can not cooperate due to mental diseases or other reasons. |

The eligibility criteria text is organized with multiple sentences. After sentence segmentation and filter, we randomly selected 19185 sentences used in the unsupervised hierarchical clustering section for criteria categories induction, and 38341 sentences used in supervised criteria classification section for the classification capacity validation of induced semantic categories.

## clustering
We used the agglomerative hierarchical clustering algorithm, works in a “bottom-up” manner, to cluster constructed semantic feature matrix and generated clusters based on criteria sentences similarity. The package ‘scikit-learn’ of Python was applied for the clustering section.

1. get 19185 Chinese-English eligibility criteria sentences pairs.
2. get UMLS semantic types of English eligibility criteria sentences using MetaMap.
3. convert criteria sentences to feature matrix based on UMLS semantic types.
4. hierarchical clustering and semantic categories induction.

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

## Contacts
Zong Hui, Tongji University, Shanghai, 200092, China, [zonghui@tongji.edu.cn](mailto:zonghui@tongji.edu.cn)

