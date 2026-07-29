ADVANCED PREDICTIVE MODELLING FOR CUSTOMER CHURN IN BANKING: A COMPARATIVE ANALYSIS OF MACHINE LEARNING AND DEEP LEARNING TECHNIQUES 

DUC LE 

Research Proposal 

JULY 2025 

## **Table of Contents** 

|**LIST OF TABLES ....................................................................................................................................................................... 3**|
|---|
|**LIST OF FIGURES ...................................................................................................................................................................... 3**|
|**LIST OF APPENDICES .............................................................................................................................................................. 3**|
|**ABSTRACT ................................................................................................................................................................................. 4**|
|**1.**<br>**BACKGROUND ................................................................................................................................................................ 4**|
|**2.**<br>**LITERATURE REVIEW .................................................................................................................................................. 7**|
|2.1.<br>MACHINELEARNING(ML) APPROACHES FORCHURNPREDICTION.................................................................................. 10|
|2.2.<br>DEEPLEARNING(DL) APPROACHES FORCHURNPREDICTION.......................................................................................... 10|
|2.3.<br>IMBALANCEHANDLING ANDPERFORMANCEMETRICS........................................................................................................ 11|
|2.4.<br>RESEARCHGAPS ANDOPPORTUNITIES................................................................................................................................... 11|
|**3.**<br>**RESEARCH QUESTIONS ............................................................................................................................................ 12**|
|**4.**<br>**AIM & OBJECTIVES ..................................................................................................................................................... 13**|
|4.1.<br>AIM: ............................................................................................................................................................................................... 13|
|4.2.<br>OBJECTIVES: ................................................................................................................................................................................. 13|
|**5.**<br>**SIGNIFICANCE OF THE STUDY ................................................................................................................................ 14**|
|**6.**<br>**SCOPE OF THE STUDY ............................................................................................................................................... 15**|
|6.1.<br>INCLUSIONS: ................................................................................................................................................................................. 15|
|6.2.<br>EXCLUSIONS: ................................................................................................................................................................................ 15|
|**7.**<br>**METHODOLOGY .......................................................................................................................................................... 16**|
|7.1.<br>DATASET DESCRIPTION.............................................................................................................................................................. 17|
|7.2.<br>DATAPREPROCESSING............................................................................................................................................................... 17|
|_Missing Data Handling .......................................................................................................................................................................... 17_|
|_Categorical Feature Encoding ............................................................................................................................................................ 19_|
|_Feature Scaling ......................................................................................................................................................................................... 20_|
|_Class Imbalance Treatment ................................................................................................................................................................. 21_|
|7.3.<br>FEATUREENGINEERING............................................................................................................................................................. 23|
|7.4.<br>TRAIN–TESTSPLIT ANDVALIDATION..................................................................................................................................... 24|
|7.5.<br>ALGORITHMSCONSIDERED........................................................................................................................................................ 25|
|_Comparative Perspective (ML vs. DL in the four-core metrics) ........................................................................................... 25_|
|_Machine Learning Models .................................................................................................................................................................... 26_|
|_Deep Learning Models ............................................................................................................................................................................ 26_|
|7.6.<br>HYPERPARAMETEROPTIMISATION.......................................................................................................................................... 27|
|_Machine Learning Models .................................................................................................................................................................... 27_|
|_Deep Learning Models ............................................................................................................................................................................ 28_|
|<br>7.7.<br>EVALUATIONMETRICS............................................................................................................................................................... 29|
|7.8.<br>INTERPRETABILITYANALYSIS................................................................................................................................................... 34|
|**8.**<br>**REQUIRED RESOURCES ............................................................................................................................................ 35**|
|**9.**<br>**RESEARCH PLAN ......................................................................................................................................................... 35**|
|**10.**<br>**LIMITATIONS ......................................................................................................................................................... 37**|
|**REFERENCES .......................................................................................................................................................................... 38**|
|**APPENDIX ............................................................................................................................................................................... 40**|



## **List of Tables** 

|TABLE1: LITERATURE REVIEW SUMMARY....................................................................................................................................................... 7|
|---|
|TABLE2: OBJECTIVES ANDRESEARCHQUESTIONS ALIGNMENT................................................................................................................ 13|
|TABLE3: COMPARISON OFMISSINGDATAIMPUTATIONMETHODS......................................................................................................... 18|
|TABLE4: COMPARISON OFENCODINGMETHODS......................................................................................................................................... 19|
|TABLE5: COMPARISON OFCLASSIMBALANCETREATMENTTECHNIQUES.............................................................................................. 21|
|TABLE6: MODEL PERFORMANCE COMPARISON............................................................................................................................................ 25|
|TABLE7: COMPARISON OFHYPERPARAMETEROPTIMISATIONAPPROACHES FORMACHINELEARNINGMODELS.......................... 27|
|TABLE8: RUNTIMEBENCHMARKINGPROTOCOL FOREXECUTIONTIME................................................................................................. 31|
|TABLE9: EVALUATIONMETRICS ANDSTATISTICALSIGNIFICANCE USINGCROSSVALIDATION........................................................... 32|
|TABLE10: TABLE OFDETAILRESEARCHPLAN............................................................................................................................................ 35|



## **List of Figures** 

|FIGURE1: RESEARCHMETHODOLOGYWORKFLOW..................................................................................................................................... 16|
|---|
|FIGURE2 - EXAMPLE OFSTANDARDISATION SCALING................................................................................................................................. 21|
|FIGURE3 - DEMONSTRATION OF STRATIFIED K-FOLDCROSS-VALIDATION............................................................................................ 24|
|FIGURE4: CONFUSION MATRIX........................................................................................................................................................................ 29|
|FIGURE5: WORKFLOW OFSTATISTICALSIGNIFICANCETEST.................................................................................................................... 33|
|FIGURE6 - RESEARCH PLAN............................................................................................................................................................................. 35|



|**List of Appendices**|
|---|



|APPENDIXA: DATASETSTRUCTURE ANDDESCRIPTION.............................................................................................................................. 40|
|---|
|APPENDIXB: PROPOSEDENGINEEREDFEATURES....................................................................................................................................... 42|



## **ABSTRACT** 

Customer churn in banking is a costly problem, as retaining existing customers is significantly more affordable than new customer acquisition. With the help of predictive analytics algorithms, banks can easily identify at-risk customer at an early stage and intervene with appropriate retention planning. Using a public Kaggle dataset of over 5,000 banking customers, this paper studies churn prediction applying both advanced machine learning (ML) and deep learning (DL) techniques. 

The methodology combines data preprocessing, model training with appropriate hyperparameter tuning, model evaluation and interpretability analysis. Data preprocessing covers missing value imputation, one hot encoding for low-cardinality categorical feature, feature scaling and class imbalance treatment using SMOTE and GAN-based method. While Logistic Regression acts as baseline, Decision Tree, Random Forest, and XGBoost are candidates for ML, Artificial Neural Networks as baseline and Deep Neural Networks are selected for DL. Hyperparameter tuning is applied with Optuna, and model performance is evaluated using accuracy, precision, recall, F1score, AUC, and execution time. 

To ensure actionable insight, the final model must be interpreted and understood correctly. It is necessary to employ interpretability analysis such as SHAP. This enables both global and local explanations of model predictions. The paper results are expected to provide a fair view of performance versus computational efficiency, guiding practitioners in figuring out the best predictive framework for real-time deployment in banking. It is noted, however, that deep learning models are not subjected to full hyperparameter optimisation due to dataset size and computational constraints, representing a conscious trade-off within the study’s scope. 

## **1. BACKGROUND** 

In banking sector, customer churn, the behavior by which clients terminate their relationship with a bank, remains one of the most pressing strategic challenges. New customer acquisition cost is approximately about five and seven times more costly than old customer retention cost thus even small retention increase can result in substantial bottom-line improvements result in substantial bottom-line improvements (Bhuria _et al_ ., 2025; Singh, H., _et al_ ., 2024). Churn has several impacts. Besides its direct impact on revenue from deposit, lending and fee-based services, it has two additional indirect losses. It is a long-term loss because the loss of cross sales opportunities. The other is the loss of brand loyalty (AbdelAziz _et al_ ., 2025; Abbas _et al_ ., 2023). 

Nowadays, due to the rise of fintech rivals, neo bank, and technology-driven service models, the competitive intensity in retail and digital banking has increased sharply. Especially, with the technology innovation, mobile banking or digital banking enables low barrier to entry, allowing customers to compare and move to alternative banks in real time (Murindanyi _et al_ ., 2025). Therefore, high value customer retention is no longer a simple marketing function, but it requires a cooperation across functions. The recent customer retention strategy usually includes analytics, operations and strategic planning (Advances in Economics, Management & Political Sciences, 2024). 

While banking has its own structural drivers of churn, similar experiences are observed in other industries. In telecommunications, because of high competition and low switching barriers, various churn prediction models ranging from logistic regression to advanced ensemble methods have been comprehensively developed and demonstrated strong performance (Boozary, 2025). Similarly, churn prediction models, based on purchase frequency, browsing behaviour, and promotional response patterns, are also well implemented in e-commerce to detect high risk of platform leaving. Those models that are trained using gradient boosting and deep neural networks algorithms, have shown particularly strong recall (Ahmed _et al._ , 2023). 

There is clear proof saying that despite the domain differences, the methodological challenges such as class imbalance, feature selection, and the trade-off between interpretability and accuracy are remarkably consistent. These shared analytical challenges have powered the search for more sophisticated and generalisable predictive frameworks that can be applied across multiple industries, including banking. 

Predictive analytics has become an essential competence for modern retention planning in other industries. At early stage, traditional statistical models such as logistic regression were used for churn prediction in banking. They were good for their interpretability yet their ability in capturing complex non-linear behaviors were very limited (AbdelAziz _et al_ ., 2025; Patel, S., _et al_ ., 2024). Thanks to the rapid evolution of machine learning (ML), ensemble methods such as random forests and gradient boosting has been introduced. They leverage multiple base learners to improve accuracy and generalization (Singh, P., _et al_ ., 2025). These models have demonstrated competitive performance in banking datasets, as well as in related sectors like telecommunications and e- commerce (Boozary, 2025). Nevertheless, when feature relations are highly complex or when predictive signals are embedded in high-dimensional, noisy, or heterogeneous data a limitation, even the most advanced ML ensembles can struggle. Hence, deep learning has emerged as a powerful approach. 

More recently, deep learning (DL) approaches have expanded the analytical capabilities by enabling the automatic extraction of complex feature interactions from high-dimensional data. Architectures such as artificial neural networks (ANNs) and deep neural networks (DNNs) have achieved notable improvements in recall and F1-score, especially when combined with appropriate feature scaling and optimization (Domingos, Ojeme & Daramola, 2021). Hybrid models that integrate convolutional neural networks (CNNs) with recurrent structures like bidirectional long short-term memory (BiLSTM) have further demonstrated strong performance in identifying subtle churn signals (Ahmed _et al_ ., 2023; Zhang _et al_ ., 2024). In comparative experiments, Singh _et al_ ., (2023) and Adamu _et al_ ., (2025) report that DL can outperform gradient boosting methods under certain configurations, although at a higher computational cost. 

Although DL performance is impressive, practical application in banking is considerably low because of two main reasons. DL models usually consume significant computational resources and longer training times and are inherently less interpretable than traditional ML (Basit _et al_ ., 2024). Especially, this interpretability challenge is getting more significant in regulated environments, where explainable AI (XAI) is becoming a compliance expectation (Tékouabou, 2022; Ma _et al_ ., 2022). By integrating interpretability techniques such as SHAP values into churn models few papers such as Li, X., _et al_ . (2025) and Singh, A., _et al_ . (2024) address this gap between predictive accuracy and regulatory transparency. 

Class imbalance, where churn customers play only a small proportion of the total customer base, is another repeated challenge in churn prediction. During training, models may be biased toward the majority class, reducing the ability to detect true churners, even when overall accuracy appears high. A range of techniques have been proposed to address this, including synthetic minority oversampling (SMOTE), adaptive synthetic sampling (ADASYN), and more recently, generative adversarial network (GAN)-based oversampling (Adiputra, Wanchai & Lin, 2025). In addition, cost-sensitive learning approaches, where misclassifications of churners incur higher penalties, have been shown to improve recall and business impact in ensemble models (Bhuria _et al_ ., 2025; Abubakar _et al_ ., 2024). 

Hyperparameter tuning plays a critical role in maximizing the potential of both ML and DL models. Although traditional methods such as grid search and random search are still common, they may be inefficient for complex models with large parameter spaces. Optuna, a more advanced framework, has been introduced to restructure the search process, applying pruning strategies to terminate underperforming trials early (Domingos, Ojeme & Daramola, 2021; Patel _et al_ ., 2023). As highlighted by Marmion _et al_ . (2012) and Molnar (2020), model interpretability is considered together with hyperparameter tuning strategies to ensure that tuning does not compromise transparency. 

Despite extensive prior work, there are few gaps remain. First, although both ML and DL have been used to churn prediction in banking, there is a lack of direct head-to-head comparisons conducted under identical experimental setups (AbdelAziz _et al_ ., 2025; Basit _et al_ ., 2024). Second, execution time is usually underestimated and so frequently overlooked as a performance metric regardless its significance for real-time inference systems (Singh, P., _et al_ ., 2025). Third, there is only a limited empirical evidence evaluating the effect of preprocessing, imbalance handling, and hyperparameter optimization combination on model performance in banking sector (Adiputra, Wanchai & Lin, 2025; Patel _et al_ ., 2023). 

By combining three actions at once, this thesis seeks to narrow these gaps. First, it evaluates both ML and DL algorithms on a public banking churn dataset. Second, it applies sophisticated sampling approaches and rigorous hyperparameter tuning. Lastly, the paper also comprehensive evaluates predictive performance and operational efficiency. The first goal of this paper is to assists practitioners in model selection that balance performance, interpretability, and deployment feasibility. Another goal is to contribute to an academic understanding of predictive modeling in financial services. 

## **2. LITERATURE REVIEW** 

Customer churn prediction is an interesting problem that has always attracted many researcher from all over the world and across all industries. Each study brings domain specific insight and methodological innovations. In banking, this problem has grown from traditional statistical approaches to advanced machine learning (ML) and deep learning (DL) architectures,  resulting from a bigger trends from other sectors such as telecommunications and ecommerce (Boozary, 2025; Ahmed _et al_ ., 2023). As stated in the background sections, these industries share common analytical challenges class imbalance, the need for robust feature selection, and the trade-off between interpretability and predictive accuracy despite differences in customer behaviour and operational contexts. As supported by the cross-industry evidence, it is reasonable for evaluating a various set of ML and DL approaches in the banking sector. Besides the motivation to benchmark performance, it is also to access their adaptability to the sector’s unique regulatory and customer engagement requirements. 

In banking, predictive analytics for churn must address the dual challenge of delivering high predictive accuracy while meeting operational, regulatory, and interpretability requirements. This section synthesizes prior work into three main themes: machine learning approach, deep learning approach, performance metrics and imbalance handling. The table below describe a high-level summary of prior literatures. 

||**Tab**<br>|**le 1: Literature rev**<br>|**iew summar**<br>|**y**<br>||
|---|---|---|---|---|---|
|**Section**|**Problem**|**Methodology**|**Dataset**<br>**Used**|**Interpretation of**<br>**Results**|**Citation**|
||Low accuracy<br>and overfitting<br>in simple<br>decision trees|Logistic<br>Regression,<br>Decision Trees|Banking<br>dataset|Logistic regression<br>provides<br>interpretability but<br>lower accuracy<br>thanensembles|Abbas_et_<br>_al_. (2023)|
||Benchmarking<br>ML vs DL for<br>churn<br>prediction|Gradient<br>Boosting,<br>Random Forest|Banking<br>dataset|Gradient boosting<br>outperforms<br>bagging ensembles<br>whentuned|AbdelAziz<br>_et al_.<br>(2025)|
|Machine<br>Learning<br>Approaches|Evaluating<br>baseline ML<br>classifiers|Logistic<br>Regression,<br>Decision Trees|Banking<br>dataset|Logistic regression<br>valued for<br>transparency, DTs<br>for interpretability|Patel, S.,<br>_et al_.<br>(2024)|
||Improving<br>prediction<br>stability|Random<br>Forest,<br>XGBoost|Banking<br>dataset|Ensembles improve<br>generalisation and<br>reduce overfitting|Singh, P.,<br>_et al_.<br>(2025)|
||Cross-industry<br>churn<br>application|Random<br>Forest,<br>boosting|Ecom<br>dataset|Boosting<br>techniques yielded<br>highest accuracy|Boozary<br>(2025)|



|Cost-sensitive<br>churn<br>classification|SVM, Logistic<br>Regression|Banking<br>dataset|SVM captures<br>nonlinear decision<br>boundaries|Abubakar<br>_et al_.<br>(2024)|
|---|---|---|---|---|
|Ensemble<br>stability in<br>churn models|Hybrid<br>Ensembles<br>(Voting,<br>Stacking)|Telco<br>dataset|Hybrid ensembles<br>yield more stable<br>predictions|Bhuria_et_<br>_al_. (2025)|
|Multi-sector<br>churn analysis|Random<br>Forest,<br>ensembles|Banking<br>&<br>Telecom<br>dataset|Ensemble methods<br>robust across<br>domains|Murindan<br>yi_et al_.<br>(2025)|
|Manual feature<br>engineering<br>limits<br>complexity|ANN|Banking<br>dataset|ANN improves<br>recall over ML<br>baselines|AbdelAziz<br>_et al_.<br>(2025)|
|Exploring DL<br>model<br>effectiveness|DNN|Banking<br>dataset|DNN achieves<br>higher F1 when<br>tuned properly|Adamu_et_<br>_al_. (2025)|
|Comparative<br>DL for churn<br>prediction|DNN|Banking<br>dataset|DNN captures<br>higher-order<br>featureinteractions|Singh, H.,<br>_et al_.<br>(2024)|
|Feature-rich<br>DL for churn|DNN|Banking<br>dataset|DNN shows<br>significant<br>accuracy gains|Basit_et al_.<br>(2024)|
|Deep Learning<br>Approaches<br>Sequential<br>dependencies<br>in churn|CNN–<br>BiLSTM<br>hybrid|Telco<br>dataset|CNN–BiLSTM<br>improves recall by<br>capturing temporal<br>dependencies|Ahmed_et_<br>_al_. (2023)|
|Hybrid<br>network for<br>churn|CCP-Net|Banking<br>dataset|CCP-Net detects<br>spatial–temporal<br>churn patterns|Zhang_et_<br>_al_. (2024)|
|Temporal<br>patterns in<br>churn|LSTM|Telco<br>dataset|LSTM improves<br>sequence<br>modelling in churn|Sezer_et_<br>_al_. (2018)|
|Complexity<br>and<br>interpretability|DL with<br>feature<br>selection|Banking<br>dataset|Feature selection<br>reduces complexity<br>and cost|Ma_et al_.<br>(2022)|
|Performance<br>High cost of<br>misclassificati<br>on|Accuracy,<br>Recall, F1-<br>score|Banking<br>dataset|Recall is most<br>critical for business<br>impact|Bhuria_et_<br>_al_. (2025)|
|Metrics<br>Benchmarking<br>metrics in<br>churn|Accuracy,<br>Precision,<br>Recall|Banking<br>dataset|Precision–recall<br>trade-off must be<br>balanced|AbdelAziz<br>_et al_.<br>(2025)|



||Emphasis on<br>churn<br>detection|Recall, F1-<br>score|Banking<br>dataset|Recall<br>improvements<br>strongly affect<br>business outcomes|Singh, P.,<br>_et al_.<br>(2025)|
|---|---|---|---|---|---|
||Metric<br>evaluation<br>across domains|Accuracy,<br>Recall, F1-<br>score|Banking<br>+<br>Telecom<br>dataset|F1 balances recall–<br>precision better<br>than accuracy|Adiputra,<br>Wanchai<br>& Lin<br>(2025)|
||Severe class<br>imbalance|SMOTE<br>oversampling|Banking<br>dataset|Improves recall by<br>increasing minority<br>samples|Tékouabo<br>u (2022)|
||Advanced<br>imbalance<br>correction|GAN-based<br>oversampling|Banking<br>dataset|GAN generates<br>realistic minority<br>samples|Ali, M.,_et_<br>_al_. (2024)|
|Imbalance<br>Handling|Data<br>imbalance in<br>churn|Cost-sensitive<br>learning|Telco<br>dataset|<br>Penalising churner<br>misclassification<br>improves recall|Imran_et_<br>_al_. (2023)|
||Hybrid<br>ensemble<br>imbalance<br>correction|RUSBoost|Banking<br>dataset|Combines under-<br>sampling +<br>boosting<br>effectively|Bhuria_et_<br>_al_. (2025)|
||Lack of<br>interpretability<br>in black-box<br>models|SHAP,<br>interpretable<br>ensembles|Banking<br>dataset|<br>SHAP improves<br>transparency with<br>minimal accuracy<br>loss|Li, X.,_et_<br>_al_. (2025)|
|ML<br>Explainability|Interpretable<br>churn<br>prediction|LIME + ML<br>models|Banking<br>dataset|LIME explanations<br>enhance trust in<br>churn predictions|Singh, A.,<br>_et al_.<br>(2024)|
||Oversampling<br>+<br>interpretability|SMOTE with<br>interpretable<br>ensembles|Banking<br>dataset|Improves recall<br>while preserving<br>interpretability|Tékouabo<br>u (2022)|
||Over-<br>optimistic<br>evaluation|Stratified k-<br>fold CV|Banking<br>dataset|Cross-validation<br>provides robust<br>estimates|Ahmed_et_<br>_al_. (2023)|
|Evaluation<br>Strategy|Statistical<br>confirmation<br>of models|Significance<br>testing|Banking<br>dataset|Validates<br>differences<br>between models|Marmion<br>_et al_.<br>(2012)|
||Interpretability<br>as complement|Explainability<br>frameworks|Banking<br>dataset|Interpretability<br>enhances trust in<br>high-performing<br>models|Molnar<br>(2020)|



### 2.1. **Machine Learning (ML) Approaches for Churn Prediction** 

For churn prediction problem, it is nature for data to be well structure in tabular format, and ML algorithms are well known for their strong performance on that kind of data. Therefore, ML has been always a wise choice for churn prediction. Each ML algorithms are well known for their characteristics. 

Logistic Regression (LR) is very good for its transparency and statistically simple (Abbas _et al_ ., 2023; AbdelAziz _et al_ ., 2025) and Support Vector Machines (SVM) is worthy for their ability to handle nonlinear decision boundaries (Abubakar _et al_ ., 2024).  Decision Tree (DT) algorithm is the best for their interpretability and rapid inference, which make it a good choice for customer retention system deployment (Bhuria _et al_ ., 2025; Patel, S., _et al_ ., 2024). Those three models traditional, well-known ML algorithms, and they are acting as bassline. However, DT usually falls behind compared to ensemble methods (Singh, P., _et al_ ., 2025). 

Random forest (RF) models, which combine multiple DTs through bagging and random feature selection, have been shown to improve accuracy and generalization while mitigating overfitting (Boozary, 2025). Gradient boosting algorithms, including Gradient Boosted Decision Trees (GBDT) and XGBoost, offer further refinements by sequentially refining weak learners (Adiputra, Wanchai & Lin, 2025; AbdelAziz _et al_ ., 2025; Li, X., _et al_ ., 2025). According to Patel _et al.,_ (2023) if boosting methods are properly tuned, they may outperform bagging ensembles.  Hybrid ensemble strategies such as voting classifiers and stacking have been applied to combine strengths from multiple algorithms, often yielding more stable predictions than any single model (Murindanyi _et al_ ., 2025). 

In practice, one of the barriers to apply ML is the difficulty to understand them. Therefore, nowadays explainability has also begun as a focus in ML-based churn models. By integrating interpretable machine learning frameworks e.g. SHAP and LIME, into banking churn prediction, Li, X., _et al_ . (2025) and Singh, A., _et al_ . (2024) successfully enable transparency without significantly sacrificing accuracy. Tékouabou (2022) proposes an approach that combines SMOTE oversampling with interpretable ensemble models to improve both performance and explainability. 

### 2.2. **Deep Learning (DL) Approaches for Churn Prediction** 

Deep learning (DL) offers the ability to automatically learn complex feature interactions, reducing the need for manual feature engineering. Artificial neural networks (ANNs) have been applied in banking churn prediction with encouraging results, often outperforming baseline ML models in recall and F1-score (AbdelAziz _et al_ ., 2025; Domingos, Ojeme & Daramola, 2021; Adamu _et al_ ., 2025). Deep neural networks (DNNs), which add multiple hidden layers to ANNs, further enhance the modelling of high-order feature interactions (Singh, H., _et al_ ., 2024; Basit _et al_ ., 2024). 

Hybrid DL architectures combine different network types to capture both spatial and temporal relationships in customer behavior. Ahmed _et al_ . (2023) propose a CNN–BiLSTM hybrid for telco churn prediction, achieving substantial recall improvements, while Zhang _et al_ . (2024) present CCP-Net, a hybrid neural network framework adaptable to banking datasets. Sezer _et al_ . (2018) demonstrate that LSTM-based architectures can capture sequential dependencies, an approach relevant for modelling transaction histories in banking. 

Recent studies are also exploring feature selection within DL pipelines to reduce model complexity and improve training efficiency. Abbas _et al_ . (2023) demonstrate the successful adaptation of DL models to banking data through domain-specific preprocessing. 

While DL models deliver strong predictive performance, they present challenges in interpretability and computational demands (Adiputra, Wanchai & Lin, 2025; Murindanyi _et al_ ., 2025). Ma _et al_ . (2022) stress the importance of interpretable AI in banking, noting that DL models require posthoc explanation methods to meet regulatory requirements. 

### 2.3. **Imbalance Handling and Performance Metrics** 

Class imbalance is a persistent problem in churn prediction across industries. There are two approaches, either data level or algorithm level. Data-level techniques such as SMOTE (Adiputra, Wanchai & Lin, 2025; Tékouabou, 2022) and GAN-based oversampling (Adiputra, Wanchai & Lin, 2025) have been used to increase minority class representation. Algorithm-level methods, including cost-sensitive learning (Bhuria _et al_ ., 2025) and probability calibration techniques (Abubakar _et al_ ., 2024), have been employed to prioritize churner detection. 

The choice of performance metrics is critical in churn prediction, where misclassifying a churner carries a greater business cost than a false positive. There are four widely used metrics, which are accuracy, precision, recall and F1-score (Adiputra, Wanchai & Lin, 2025; Bhuria _et al_ ., 2025). In banking churn prediction, among those four metrics, recall is exclusively key. Because retaining an at-risk customer can return higher long-term value (AbdelAziz _et al_ ., 2025; Singh, P., _et al_ ., 2025). 

Evaluation strategies normally involve k-fold cross-validation to ensure robust performance estimates. Ahmed _et al_ . (2023) and Marmion _et al_ . (2012) support for statistical significance testing to validate observed differences between models, while Molnar (2020) emphasizes interpretability as an essential complement to predictive metrics. 

### 2.4. **Research Gaps and Opportunities** 

After studying many papers, there are still some gaps needed to be deeply investigated and resolved. First, it is quite rare to find comparative studies, which evaluate ML and DL models under identical banking domain conditions (AbdelAziz _et al_ ., 2025; Basit _et al_ ., 2024). 

Second, although execution time is an operationally critical factor for real-time interventions, it is usually ignored from and evaluation always focuses on predictive performance (Singh, P., _et al_ ., 2025). 

Third, preprocessing, imbalance handling, and hyperparameter tuning are often applied in isolation rather than systematically combined and evaluated for their cumulative effect on performance (Adiputra, Wanchai & Lin, 2025; Patel _et al_ ., 2023). The adoption of advanced optimization frameworks such as Optuna remains limited in banking churn research, despite evidence of its efficiency (Domingos, Ojeme & Daramola, 2021). 

Finally, besides predictive performance and execution time, the result explainability is also very important and should be taken with enough care. Studies like Li, X., _et al_ . (2025), Singh, A., _et al_ . (2024), and Ma _et al_ . (2022) point to the need for churn prediction systems that are both accurate and interpretable, aligning with regulatory requirements and enhancing stakeholder trust. 

## **3. RESEARCH QUESTIONS** 

This research is guided by four central questions derived from both literature gaps and the operational needs of banking institutions. 

First, it presents a head-to-head comparison of how selected machine learning and deep learning models perform in churn prediction in banking sector under a reliable experimental basis. Despite there are few papers evaluate these models individually (AbdelAziz _et al_ ., 2025; Singh, H., _et al_ ., 2024; Adamu _et al_ ., 2025), few offer direct comparative analysis within the same data and processing conditions. 

Second, the essential requirement of go-to-market churn prediction in real time retention system in banking sector is computational execution time must be quick. Although execution time is crucial factor, it is usually underestimated compared to predictive performance according to Singh, P., _et al_ . (2025). This paper investigates the trade-off between those two factors and to re-evaluate the role of ensemble methods. 

Third, it studies how class imbalance handling approaches and feature engineering affects churn prediction performance in banking industry. It is believed that those techniques have a considerably impact on recall and F1-score (Adiputra, Wanchai & Lin, 2025; Tékouabou, 2022; Adiputra, Wanchai & Lin, 2025). 

Besides class imbalance handling method, hyperparameter tuning can extensively increase predictive accuracy yet it is rarely assessed across model families within a unified experimental setting (Domingos, Ojeme & Daramola, 2021; Patel _et al._ , 2023). The next question investigates the usefulness of different hyperparameter tuning methods for both ML and DL models. 

Despite how well predictive models perform, they are still none-senses to business stake holders if they don’t understand them. The final question address how interpretability analysis help to gain trust from entrepreneurs. 

#### **In short, this research addresses:** 

- How machine learning and deep learning models perform in churn prediction across multiple performance metrics, in particular banking industry. 

- How predictive performance trades off with execution time for real-time deployment. 

- How preprocessing and imbalance-handling techniques influence model performance. 

- How different hyperparameter optimisation strategies perform across ML and DL models. 

- How interpretability analysis brings more understanding to stake holders. 

## **4. AIM & OBJECTIVES** 

### 4.1. **Aim:** 

This paper’s primary purpose is to carefully evaluate advanced predictive models for customer attrition in the banking sector, using both traditional machine learning algorithms and deep learning architectures. By conducting a methodical comparative analysis, the study aims to identify more effective solutions in terms of predictive performance, computational efficiency, and interpretability thereby offering actionable insights for improving customer retention strategies. 

### 4.2. **Objectives:** 

- i. To review existing literature on churn prediction and identify practical gaps, particularly in comparing machine learning and deep learning algorithms in banking. 

- ii. To implement and evaluate machine learning and deep learning models on a common banking dataset. The performance is evaluated across multiple metrics. 

- iii. To analyse the trade-off between predictive performance and execution time, as well as the impact of preprocessing and class-imbalance treatments on model. 

- iv. To assess the effectiveness of different hyperparameter tuning methods in enhancing both machine learning and deep learning models and derive useful insights for deployment in banking sectors. 

- v. To explain the difficulty in predictive models to typical executives. 

The table below sums up the objectives and research questions alignment. 

|**Table 2: Objectives and Re**<br>|**search Questions alignment**<br>|
|---|---|
|**Objectives**|**Research questions**|
|Review literature & identify gaps (ML vs DL<br>inbanking churn)|How do ML and DL models compare in<br>predicting banking churn acrossmetrics?|
|Implement ML & DL models and benchmark<br>across metrics|How does predictive performance trade off<br>with execution time in real-time deployment?|
|Investigate performance-execution time trade-<br>off & preprocessing impact|How do preprocessing and imbalance-handling<br>techniques influence model performance?|
|Evaluate<br>hyperparameter<br>optimisation<br>strategies for ML & DL|How do different hyperparameter optimisation<br>strategies perform across ML and DL models?|
|Explain predictive model to business people.|How does interpretability analysis bring more<br>understanding to stake holders?|



## **5. SIGNIFICANCE OF THE STUDY** 

This study offers banks and financial institutes an analytical recommendation on the proper predictive model application for customer retention planning. Even though predictive analytics has certain reputation in customer retention actions, it is scarce to see a head-to-head comparison of machine learning and deep learning techniques on banking specific datasets (AbdelAziz _et al_ ., 2025; Singh, H., _et al_ ., 2024). Given that absence of comparison research, banking customer retention strategies are often determined by outcome from many different settings, which may not reflect standard behaviors in financial services specific to the financial services sector. 

As discussed earlier in background sections, there are some gaps between previous literature and expectation. This study seeks to narrow those gaps. First, it brings both machine learning and deep learning model under identical experimental settings, which is same datasets for the same industry. Second, it also evaluates those algorithms in term of predictive performance which are accuracy, precision, recall, F1-score, AUC and execution time. Those metrics are discussed detailly in methodology section. Execution time is often underestimated in model section studies. It is carefully evaluated in this paper, because it is essentially efficiency for real time retention systems, where milliseconds may result in wrong decision making (Singh, P., _et al_ ., 2025). 

According to prior works, by integrating modern imbalance treatment and hyperparameter tuning can remarkably increase model sensitivity to churn (Ali, M., _et al_ . 2024, Imran _et al_ . 2023, and Bhuria _et al_ . 2025). Hence, this study tries to apply this concept across several machine learning and deep learning in banking specific dataset. This study aims to provide a practical guidance data science teams aiming to optimize returns on analytics investments. 

In banking sectors, banking practitioners are often cautious and may distrust models they cannot understand. Hence, this paper also aims to discuss explainable AI (XAI), which is said that it may help to increase interpretability. According to Li, X., _et al_ . (2025), Singh, A., _et al_ . (2024), and Ma _et al_ . (2022), the interpretability frameworks promises that study findings are interpreted in such a way that ordinary people may grasp the underlying rationale. 

This work enhances academic knowledge and industrial practice by providing a validated and replicable methodology for predicting turnover in banking. The insights will assist decision-makers in choosing models that perform well in offline evaluations while meeting the operational, regulatory, and strategic requirements of live banking environments. This puts predictive analytics as an essential facilitator of sustainable customer relationship management amid increased competition and digital change. 

## **6. SCOPE OF THE STUDY** 

### 6.1. Inclusions: 

- Analysis is limited to the provided Kaggle banking churn dataset, consisting of over 5,000 customer records with demographic, behavioural, and account-related attributes. 

- Models evaluated will be confined to the specified machine learning and deep learning algorithms. The list of algorithms is described more detail in methodology section. 

- Evaluation will be conducted using six metrics: accuracy, precision, recall, and F1-score, AUC and execution time. Those metrics and selection rationale are explained detailly in methodology section. 

- Machine learning hyperparameter tuning will be performed using Grid Search, Random Search and Optuna. Deep learning hyperparameter will be set using prior study setups. 

### 6.2. Exclusions: 

- Real-time data streaming architectures and machine learning ops for continuous integration/continuous development (CI/CD). 

- Customer churn prediction in non-banking domains. 

- Other deep learning architectures such as convolutional or recurrent networks are not part of this study, because of computational constraints. 

- Full hyperparameter tuning for deep learning model is excluded from this study. While machine learning models are properly tuned, deep learning models are using previous literature configurations. It may reflect a conscious trade-off between fairness, computational feasibility and scope. 

## 7. Methodology
### 7.1. Dataset description 

This study uses the public dataset from Kaggle, which is customer_churn_1M.csv. The dataset has 1 millions rows, and it is one row per customer. It does not contain historical transactions, it contains the latest related insights required to understand why customer might leave and how to retain them. 

### 7.2. Data Preprocessing 

Since the quality and structure of input data have a direct impact on the effectiveness and reliability of ML and DL models, data preparation is an essential step in predictive modeling (Patel _et al_ ., 2023; Adiputra, Wanchai & Lin, 2025). According to previous literature in churn prediction research, the data preprocessing should include missing value imputation, categorical encoding, feature scaling, class imbalance handling (Bhuria _et al_ ., 2025; Adiputra, Wanchai & Lin, 2025). 

#### **Missing Data Handling** 

Missing values data has some impact to final model training. Either it may bias the model or reduce the effective sample size. It results in poor predictive reliability (Abbas _et al_ ., 2023; Abubakar _et al_ ., 2024). Thus, it must be handled properly. There are few techniques available in prior works. First, mean imputation, median imputation or mode replaces the missing value by mean, median or mode of the feature respectively. This technique is computationally simple but highly sensitive to outliers. Using it without proper care may distort feature distribution. 

Next, regression imputation and multiple imputation by chained equations (MICE) replace missing value by an approximately statically estimation. There are some advantages and disadvantages. On the good side, they leverage relationships between features which is sometime benefit to the model training. On the other hand, they require additional computational resources and often assume unnecessary predictor linearity. 

Finally, k-nearest neighbour (kNN) replaces missing values by similar samples, which is believed to provide more sophisticated result. This inference consumes intensively computational resources on large dataset, hence scalability in banking churn prediction may be a full stop. 

The table below summarises common missing value treatment as described above. 

**<u>Table 3: Comparison of Missing Data Imputation Methods</u>** 

|**Technique**|<br>**Advantages**|<br>**Limitations**|<br>**Computational**<br>**Cost**|**Citation**|
|---|---|---|---|---|
|**Mean**<br>**Imputation**|Simple, fast|Distorted<br>by<br>outliers;<br>weak<br>distributional<br>fidelity|Very low|Abbas_et al_.<br>(2023)|
|**Median**<br>**Imputation**|Robust to outliers;<br>preserves<br>central<br>tendency|May ignore variance<br>structure|Low|Abbas_et al_.<br>(2023)|
|**Mode**<br>**/**<br>**Unknown**<br>**Class**|Suitable<br>for<br>categorical<br>data;<br>preserves absence<br>patterns|May<br>inflate<br>frequency<br>of<br>dominant class|Low|Abubakar<br>_et_<br>_al_. (2024)|
|**Regression**<br>**Imputation**|Leverages<br>relationships<br>between features|Assumes linearity;<br>risk of overfitting|Medium|Murindanyi_et_<br>_al_. (2025)|
|**MICE**<br>**(Multiple**<br>**Imputation)**|Statistically<br>principled; handles<br>uncertainty|Computationally<br>expensive; complex<br>to implement|High|Domingos,<br>Ojeme<br>&<br>Daramola<br>(2021)|
|**kNN**<br>**Imputation**|Adaptive;<br>uses<br>neighbourhood<br>similarity|Scales poorly with<br>dataset size|High|Murindanyi_et_<br>_al_. (2025)|



This study not only focus on predictive performance but also computational efficiency. Therefore, median and mode imputation methods are applied. Numerical features (e.g., balances, transaction volumes) will be imputed using the median, which mitigates the impacts of extreme outliers compared to mean replacement. Categorical features (e.g., customer demographics) will be imputed with the mode or encoded into an explicit _“Unknown”_ class. It preserves missing-data patterns as well as maintain feature interpretability. This approach ensures methodological consistency with prior banking churn studies (Abbas _et al_ ., 2023; Abubakar _et al_ ., 2024), while avoiding the excessive computational cost of more complex imputation frameworks. 

#### **Categorical Feature Encoding** 

Both ML and DL algorithms only accept numerical features, therefore it is a mandatory to transform all categorical to numerical format. There are a few encoding techniques as described in table below. 

|||**Table 4: Compariso**<br>|**n of Encoding Met**<br>|**hods**<br>|||
|---|---|---|---|---|---|---|
|**Encoding**<br>**Method**|**Problem**<br>**Addressed**|**Pros**<br>|**Cons**<br>|**Suitabilit**<br>|**y**<br>|**Citation**<br>|
|**Label**<br>**Encoding**|Converts<br>categories<br>into<br>integers|Simple, compact<br>representation|•<br>Introduces<br>artificial<br>ordinality<br>•<br>Misleads<br>algorithms<br>sensitive to<br>scale|Useful<br>tree-based<br>models<br>RF, XGB|for<br> <br>(DT,<br>oost)|Bhatia_et al_.<br>(2023)|
|**Target**<br>**Encoding**|Encodes<br>categories<br>by<br>their<br>statistical<br>relationship<br>to<br>churn<br>(e.g., mean<br>churn rate)|•<br>Captures<br>category–<br>target<br>correlation<br>•<br>Reduces<br>dimensionality|•<br>Risk<br>of<br>target<br>leakage<br>•<br>Requires<br>careful<br>cross-<br>validation|Suitable<br>high-<br>cardinalit<br>categorica<br>features|for<br>y<br>l|Abbas_et al_.<br>(2023)|
|**One-Hot**<br>**Encoding**|Expands<br>categories<br>into binary<br>indicator<br>variables|•<br>Prevents<br>artificial<br>ordinality<br>•<br>Interpretable<br>•<br>effective<br>for<br>low-<br>cardinality<br>data|Increases<br>dimensionality<br>if features have<br>high<br>cardinalities|Best for<br>cardinalit<br>attributes<br>demograp<br>in<br>M<br>models|low-<br>y<br>(e.g.,<br>hics)<br>L/DL|•<br>Boozary<br>(2025)<br>•<br>AbdelAziz<br>_et_<br>_al_.<br>(2025)<br>•<br>Patel, S.,<br>_et_<br>_al_.<br>(2024)|



The dataset includes five categorical demographic attributes, and due to low cardinalities of each features, one-hot encoding is particularly suitable method. Low cardinalities also ensure one-hot encoding computationally efficient and less prone to sparsity issues compared to high-cardinality cases. In addition, it also avoids the risk of introducing misleading ordinal relationships that could occur with label encoding, ensuring the encoded features align with the nature of the data 

One-hot encoding creates a separate binary feature (0 or 1) for each category level, ensuring that ML and DL models can process categorical data without imposing artificial ordinal relationships where none exist (Boozary, 2025). 

Formally, for a categorical variable C with k distinct categories {c1, c2, ..., ck}, one-hot encoding maps each observation x Î _C_ to a binary vector v Î{0,1}<sup>k</sup> where: 



For example, in dataset, the gender feature has 3 values Male, Female and Other. One-hot encoding will represent these as: 

|**Gender **|**Male **|**Female **|**Other **|
|---|---|---|---|
|**Male**|1|0|0|
|**Female**|0|1|0|
|**Other**|0|0|1|



In practice, we do not really need all three columns. Two columns suffice to derive the last columns. By removing one column, we also avoid multicollinearity. 

The binary churn indicator will be retained in its original 0/1 form, following conventions in prior banking churn studies (AbdelAziz _et al._ , 2025). 

#### **Feature Scaling** 

Feature scaling is an important preprocessing step for all binary classification. Especially with banking datasets, numerical features (e.g. account balances, transaction volume or transaction frequency, tenor) are in various ranges. If this step is not handled properly or frequently overlooked, larger numeric scales can dominate distance-based or gradient-based algorithms. As a result, algorithms relying on distances or gradient can suffer from slow or unstable training. In the worst case, it may also lead to biased learning and unstable convergence (Ali, S., _et al_ ., 2025). 

Hence, for those predictive algorithms which are sensitive to feature magnitude, it is a mandatory requirement to ensure uniform feature contribution. There are standardisation and normalisastion scaling methods. Normalisation scale data to a specific range which is between 0 and 1. It can distort the underlying data distribution and casually ignore meaningful patterns, especially when outliers are present. Standardisation technique rescales feature to have a mean of zero and standard deviation of one. It preserves the distribution shape while centering and scaling the data, ensuring that no single variable dominates due to its magnitude. 

Standardisation is particularly important for gradient-based models such as deep neural networks (DNNs), where unscaled features can lead to inefficient convergence (Domingos, Ojeme & Daramola, 2021), and for distance-based algorithms, where differences in scale can distort similarity measures. Standardisation is mathematically defined as: 



where x is the original feature value, 𝜇 is the feature mean, and 𝜎 is the feature standard deviation. 


|**GAN-**<br>**based**<br>**Oversampl**<br>**ing**|Uses<br>a<br>generator–<br>discriminator<br>framework to<br>produce<br>realistic<br>minority<br>samples<br><br>|•<br>•|Captures<br>complex<br>feature<br>relationships<br>Generates<br>highly<br>realistic<br>churner<br>samples|•<br>Computatio<br>nally<br>expensive<br>•<br>Risk<br>of<br>mode<br>collapse<br>in<br>GAN<br>training|<br> <br>High|Deep<br>learning|Adiputra,<br>Wanchai & Lin<br>(2025)|
|---|---|---|---|---|---|---|---|



<u>Synthetic Minority Oversampling Technique (SMOTE)</u> **:** SMOTE generates synthetic churn samples by interpolating between existing minority class instances. For each minority sample xi, one of its k-nearest neighbours xnn is selected, and a new synthetic point xnew is created as: 

𝑥"#$ = 𝑥! + 𝜆 . (𝑥"" − 𝑥!) , 𝜆 ~ 𝑈(0,1) 

Where: 

- xi: a randomly chosen minority class sample 

- xnn: one of the k-nearest neighbours of xi from the minority class 

- 𝜆: a random number drawn from a uniform distribution between 0 and 1 (U(0,1)) 

- xnew: the newly generated synthetic sample 

This interpolation along the feature space line segment between xi and xnn avoids direct duplication while enriching the minority class distribution. Studies (Adiputra, Wanchai & Lin, 2025; Tékouabou, 2022) confirm that SMOTE improves sensitivity to churners while reducing overfitting risks associated with random oversampling. 

<u>GAN-based Oversampling</u> **<u>:</u>** Generative Adversarial Networks (GANs) offer a more sophisticated approach by learning the underlying data distribution of the minority class. A GAN consists of two neural networks: the generator G(z), which maps random noise z ~ p(z) to synthetic samples, and the discriminator D(x), which attempts to distinguish real from synthetic samples. 

Through this min–max optimisation, G learns to generate synthetic churner samples indistinguishable from real ones, thereby capturing high-order correlations and producing more realistic minority examples. GAN-based oversampling has been shown to outperform interpolation-based methods in datasets with complex feature relationships (Adiputra, Wanchai & Lin, 2025). 

### 7.3. Feature Engineering 

During exploratory data analysis process new features may be derived to capture hidden customer behavior leading to churn signals. There is a clear proof from prior literature that those derived features improving discriminative power in churn models by revealing patterns not directly observable in the raw data (Singh, H., _et al._ , 2024; Abbas _et al._ , 2023). According to previous studies and banking domain expertise, five groups of derived features are proposed (a full summary of derived features is provided in appendix B). 

- a. **Tenure-related features** . The customer’s tenure, also known as month-on-book (MOB), is the month difference between account opening date and current data date. This feature provides a strong indicator of customer loyalty. A short tenure customers tend to be more volatile while longer tenure customers are more likely loyal (AbdelAziz _et al_ ., 2025). Similarly, the inactivity period, measured as the number of days since the last transaction, is another indicator for exiting, as longer inactivity period are strongly correlated with churn (Ahmed _et al_ ., 2023). 

- b. **Transaction-based features** . Metrics such as transaction frequency (number of transactions normalised by tenure), and average transaction size provide insights into how actively a customer uses their account. A balance volatility index, calculated as the absolute change in balance relative to the current balance, reflects financial instability and has been associated with higher churn tendencies. 

- c. **Financial health indicators** . A loan-to-income ratio will be engineered to assess leverage risk, complementing credit score information. Customers with higher ratios may face repayment stress, which increases the likelihood of discontinuing banking relationships. 

- d. **Service and satisfaction related** . A complaint-to-interaction ratio captures the density of dissatisfaction signals by dividing recent complaints by total customer service interactions. This is complemented by a satisfaction gap indicator, combining customer satisfaction score and complaints: low satisfaction combined with frequent complaints is a strong churn predictor, while high satisfaction and no complaints indicate stability. 

- e. **Contextual features** such as branch or region-level churn rates (derived from the training data) will be encoded as risk-based features, as certain regions or branches may consistently show higher attrition. A composite engagement index, integrating transaction frequency, satisfaction, and complaint ratios, will also be constructed to provide a holistic measure of customer commitment. 

Completion of data preprocessing and feature engineering, the dataset will be ready and optimised for both ML and DL models. Those two steps ensure selected model evaluation exposes actual algorithm capability rather than data quality or representation. 

### 7.5. Algorithms Considered 

There are two objectives behind the model selection in this study: 

- Ensuring covering of various linear classifiers, decision-tree-based ensembles, and nonlinear representation algorithms. 

- Enabling a balanced comparison between ML and DL approaches using consistent preprocessing and evaluation pipelines. 

This ensures the experimental outcomes are both statistically rigorous and operationally meaningful in a banking context. 

#### **Comparative Perspective (ML vs. DL in the four-core metrics)** 

Evaluation metrics will be discussed furtherly in the evaluation metric part in this section. Evidence from the reviewed literature consistently reveals complementary strengths between ML and DL in banking churn prediction: 

**<u>Table 6: Model performance comparison</u>** 

|**Metric**|**Best-performing**<br>**ML models**|**Best-perf**<br>**DL mode**|**orming**<br>**ls**|**Observed trends in banking churn**<br>**prediction **|
|---|---|---|---|---|
|**Accuracy**|XGBoost,<br>Random Forest|Deep<br>Network (|Neural<br>tuned)|•<br>Ensembles often lead due to bias–<br>variance optimisation<br>•<br>DNN can match with careful tuning.|
|**Precision**|Random<br>Forest,<br>Logistic Regression|Artificial<br>Network|Neural|•<br>ML ensembles typically produce<br>fewer false positives<br>•<br>ANN<br>can<br>offer<br>competitive<br>precision with feature scaling.|
|**Recall**|XGBoost<br>(with rebalancing)|Deep<br>Network|Neural|•<br>DNN<br>often<br>excels<br>in<br>recall,<br>especially with dropout and class<br>weighting<br>•<br>Important<br>for<br>catching<br>true<br>churners.|
|**F1-score**|XGBoost|Deep<br>Network|Neural|•<br>Context-dependent:<br>boosting<br>models lead on clean, separable data<br>•<br>DNN can surpass with complex<br>feature interactions.|
|**Execution**<br>**time**|Logistic Regression,<br>Decision Tree|Artificial<br>Network|Neural|•<br>Simpler models faster<br>•<br>ANN reasonable latency|



This comparative evidence motivates the inclusion of models that mutually span the metric range: ensembles (RF, XGBoost) for accuracy and precision, and DL models (DNN, ANN) for recall and balanced F1 performance. Logistic Regression and Decision Trees serve as low-latency baselines and interpretable respectively, enabling a full assessment of performance versus execution time. 

#### **Machine Learning Models** 

- **Logistic Regression (LR).** A linear probabilistic classifier that models the log-odds of churn as a function of customer attributes. LR remains a robust baseline in banking churn research due to its transparency, calibration friendliness, and low computational cost (Abbas _et al._ , 2023; AbdelAziz _et al._ , 2025; Patel _et al._ , 2024). 

- **Decision Tree (DT).** A hierarchical partitioning of the feature space that yields human-readable rules and very fast inference. While single trees often trail ensembles in raw accuracy, their latency and interpretability are attractive for real-time scoring and compliance-centric environments (Bhuria _et al.,_ 2025; Tékouabou, 2022). 

- **Random Forest (RF).** An ensemble of decorrelated trees trained via bagging and random feature selection. Therefore, RF typically improves generalisation and stability relative to a single DT. On the other hand, RF requires way more computational resources and so longer execution time than DT. Since this study also values execution time and computational resources, DT and RF are evaluated head-to-head to consider the trade-off between those execution time and predictive performance (Bhuria _et al._ , 2025; Singh, P., _et al._ , 2025). 

- **XGBoost.** Sequential ensembles that correct residual errors of prior learners and incorporate regularisation; they have repeatedly achieved state-of-the-art results on structured banking data, especially when hyperparameters are tuned systematically (Patel _et al.,_ 2023; Ali, M., _et al.,_ 2024; AbdelAziz _et al.,_ 2025; Li, X., _et al.,_ 2025). In previous literature as summarized in table above, XGBoost beats all other three machine learning model in term of accuracy, recall and F1-score. Like RF, this model also requires more computational cost. It is also good to consider the trade-off between predictive performance and computational cost. 

#### **Deep Learning Models** 

- **Artificial Neural Network (ANN):** Feed-forward multi-layer perceptions modelling nonlinear relationships between demographic, behavioural, and transactional features and churn likelihood. Typical configurations in banking churn tasks include 1–2 hidden layers with ReLU activation, dropout for regularisation, and a sigmoid output neuron for binary classification (AbdelAziz _et al_ ., 2025; Adamu _et al_ ., 2025). Like LR in machine learning model, this algorithm acts as baseline in churn prediction for deep learning. 

- **Deep Neural Network (DNN):** Extends the ANN by incorporating additional hidden layers (e.g., 4–6) to capture higher-order feature interactions in high-dimensional data. Common designs employ dense layers with ReLU activation, batch normalisation to stabilise training, dropout to mitigate overfitting, and a sigmoid output layer (Singh, H., _et al_ ., 2024; Basit _et al_ ., 2024). Because of complex architecture, DNN wins most predictive performance, yet it consumes more resources. In term of trade-off consideration, both DNN and ANN are put on the table for consideration. 

### 7.6. Hyperparameter Optimisation 

In all predictive algorithms, there are two types of parameters. First, it is model parameters such as regression coefficients or neural network weights learning and updating during the training using data. The other is model hyperparameters controlling learning process itself rather than learning from data. Some examples include the regularisation strength (e.g., L1 or L2 penalty) for logistic regression or maximum tree depth, minimum samples per split for decision tree. They are set prior to training and directly influence model capacity, convergence behaviour, and generalisation ability (Patel _et al_ ., 2023; Abubakar _et al_ ., 2024). 

#### **Machine Learning Models** 

Previous literatures believe the importance of hyperparameter tuning for maximizing the predictive performance of both ML and DL models. In contrast, deficient configurations can lead to either underfitting or overfitting. This study applies comparative approach using grid search, random search, and Optuna. 

- **Grid Search:** This method brute forces all possible combinations of predefined hyperparameters. By covering all possibilities, it can determine the best hyperparameter combination. Once the parameter space grows, the good application becomes computational burden (Abubakar _et al_ ., 2024). For example, tuning three parameters with ten candidate values each, formally ten power of three, already requires 1,000 model fits, making grid search infeasible for complex DL architectures. 

- **Random Search:** Instead of testing all combinations, random search randomly selects some of hyperparameter configurations. Obviously, it performs faster than grid search and reduces computational resources. Nonetheless, it may still spend trials on nonpotential regions of parameter spaces and lacks adaptive exploration (Abubakar _et al_ ., 2024) or it may miss the best combination. 

- **Optuna:** A modern hyperparameter optimization framework that improves on both methods by employing Bayesian optimization and early pruning strategies (Patel _et al_ ., 2023). Bayesian optimization uses probabilistic models to guide the search towards potential regions of the parameter space. On the other hand, pruning stops disappointing trials early, reallocate resources to more potential candidates. This combination returns greater efficiency in exploring complex search spaces, particularly for deep neural networks where training iterations are computationally expensive. 

**<u>Table 7: Comparison of Hyperparameter Optimisation Approaches for Machine Learning Models</u>** 

|<br>**Approach**|<br>**Description**|<br>**Strengths**|<br>**Weaknesses**|<br>**Relevant**<br>**Studies**|
|---|---|---|---|---|
|**Grid**<br>**Search**|Exhaustively<br>evaluates<br>all<br>parameter<br>combinations within a<br>predefined grid.|Simple<br>to<br>implement;<br>guarantees finding<br>best model within<br>the grid.|Computationally<br>expensive; infeasible<br>for large search spaces.|Abubakar<br>_et_<br>_al_.<br>(2024)|



|**Random**<br>**Search**|Samples<br>values<br>from<br>distributi|parameter<br>randomly<br>defined<br>ons.|More efficient than<br>grid search in high-<br>dimensional spaces;<br>can<br>reach<br>near-<br>optimal<br>results<br>quickly.|May<br>miss<br>optimal<br>combinations if not<br>sampled; results can<br>vary by run.|Patel_et al_.<br>(2023);<br>Abubakar_et_<br>_al_. (2024)|
|---|---|---|---|---|---|
|**Optuna**|Uses<br>optimisat<br>pruning<br>search<br>promisin|Bayesian<br>ion<br>and<br>to guide<br>toward<br>g regions.|Highly efficient in<br>large,<br>complex<br>spaces;<br>adaptive<br>search; early stopping<br>of poor trials saves<br>computation.|More<br>complex<br>implementation;<br>requires<br>tuning<br>of<br>optimisation process<br>itself.|Patel_et al_.<br>(2023)|



In conclusion, logically Optuna beats Grid Search and Random Search not only in term of computational resources but also achieving higher chance of finding near optimal. By applying all three approaches and comparing each approach’s peak performance with base models and them together, the study seeks the solid proof for the best tuning platform. 

#### **Deep Learning Models** 

In contrast, DL models in this study will not undergo full hyperparameter tuning process. Due to deep learning complex architectures, hyperparameter tuning requires extremely high computational resources. Their hyperparameters such as number of hidden layers, activation functions, dropout, batch size, learning rate will be set using configurations established in prior churn prediction studies (Amal _et al_ ., 2023; Ahmed _et al_ ., 2023; Zhang _et al_ ., 2024). This reflects two concerns: 

- Because the study dataset is quite small (5000 observations) and it may limit the extensive hyperparameter tuning benefit. Any overly complex tuning may risk to overfitting. 

- Full hyperparameter tuning is computationally expensive and beyond the practical scope of this study, as the scop 

By adopting configurations from previous literature, DL models are not left untuned but are setup using empirically validated settings. This ensures methodological fairness, ML models are fully tuned with Grid Search, Random Search, and Optuna, while DL models are reasonably configured to reflect their potential under realistic study constraints.  


- c. **Recall,** also known as true positive rate (TPR) or sensitivity, quantifies a model’s ability to identify relevant instances (true positives) within a dataset. It represents the proportion of actual positive cases correctly detected by the model, indicating the fraction of positives found among all actual positives. 



Recall is given particular emphasis in banking churn contexts, as failing to identify an atrisk customer may result in direct revenue loss (Tékouabou, 2022; Adiputra, Wanchai & Lin, 2025). While precision focuses on the accuracy of positive predictions, recall focuses on identifying all positives. A high recall can sometimes come at the cost of lower precision (more false positives), and vice versa. The relationship between precision and recall is often described as a trade-off. 

- d. **F1-score** provides a balanced measure of a model's performance in classification tasks, particularly useful when dealing with imbalanced datasets. It is the harmonic mean of precision and recall, offering a single value that considers both false positives and false negatives: 



The F1-score provides a single, comparable metric for evaluating and comparing the performance of different machine learning models, especially when precision and recall are both important. A high F1-score (close to 1) signifies that the model has outstanding precision and recall, meaning it is effective at both identifying positive cases and minimizing incorrect positive predictions. A low F1-score (close to 0) indicates that the model is either not identifying enough positive cases (low recall) or is making too many incorrect positive predictions (low precision), or both. In banking, either failing to retain a valuable customer or offering unnecessary incentives entail substantial cost. 

- e. **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)** : a thresholdindependent measure representing the probability that the model ranks a randomly chosen churner higher than a randomly chosen non-churner. The ROC curve plots the true positive rate (recall) against the false positive rate at various classification thresholds, and the AUC quantifies the overall separability. In general classification problems, higher AUC values indicate better discrimination between classes. In churn prediction, AUC is especially valuable because it evaluates performance across all possible thresholds, helping 

practitioners select the most appropriate operational cut-off point for intervention strategies (Ahmed _et al_ ., 2023; Patel _et al_ ., 2023). 

- f. **Execution Time** measured in seconds or milliseconds per inference or training epoch, depending on model type. It captures computational efficiency during both training and prediction phases. Built-in profiling tools provide precise measurement of both CPU and GPU operations. Execution time will be calculated by time differences between wallclock time at beginning and end of each training run or predicting. To be fair, time both training and predicting time execution will be normalized to per-epoch or per-sample 

|𝑇𝑖|=<sup>𝑒𝑛𝑑𝑡𝑖𝑚𝑒−𝑠𝑡𝑎𝑟𝑡𝑖𝑚𝑒</sup><br>𝑃𝑑=<sup>𝑒𝑛𝑑𝑡𝑖𝑚𝑒−𝑠𝑡𝑎𝑟𝑡𝑖𝑚𝑒</sup>|
|---|---|
|𝑟𝑎𝑛%|!&#  <br>𝑡𝑟𝑎𝑖𝑛𝑖𝑛𝑔 𝑠𝑒𝑡<br>𝑟𝑒%!&#  <br>𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑛𝑔 𝑠𝑒𝑡|
||**Table 8: Runtime Benchmarking Protocol for Execution Time**|
|**Aspect **|**Description **|
|**Measurement**<br>**primitives**|• For ML models: use high-resolution wall-clock timers (e.g., Python<br>time.perf_counter()).<br>• For DL models running on GPU: use CUDA synchronizers<br>(torch.cuda.synchronize()) to account for asynchronous kernel<br>execution.|
|**Warm-up &**<br>**repetitions**|• Perform 1–3 warm-up runs to populate caches and JIT compile graphs.<br>• Execute 10 independent timed runs.<br>• Report median (p50) and optionally p90 latency.<br>• Outliers are logged and reported separately.|
|**What is**<br>**timed**|• Training: fit() call (ML) or forward+backward passes (DL).<br>• Inference: predict() (ML) or forward pass (DL).<br>• Include preprocessing/transform steps required inside pipeline.|
|**What is not**<br>**timed**|• Disk I/O, dataset loading, or caching operations.<br>• One-off model/graph compilation (outside warm-up).<br>• Hyperparameter search orchestration overhead.|
|**Hardware**<br>**specification**|• Record CPU/GPU model, RAM, and storage configuration.<br>• Note whether execution uses single-thread, multi-thread, or GPU<br>acceleration.|
|**Batch size &**<br>**concurrency**|• Report latency for single-sample (online scoring) and batch (mini-<br>batch) inference.<br>• Vary concurrency levels (1, 4, 8 workers) to simulate deployment load.|
|**Reporting**|• Present results in milliseconds per inference (prediction) and seconds<br>per epoch (training).<br>• Provide confidence intervals (95%) to demonstrate stability of results.|



While more complex architectures (e.g., deep neural networks) may deliver marginal returns in predictive metrics, they often incur substantial delays in prediction speed and higher hardware costs. In churn prediction, where timeliness can influence retention intervention strategies, execution time becomes a crucial operational metric. 

   - Fast models (e.g., Logistic Regression, Decision Tree) can be deployed easily and score lower in complexity but may sacrifice some accuracy. 

   - The inclusion of execution time allows decision-makers to balance predictive power with operational feasibility. 

- g. **Evaluation Metrics and Statistical Significance** : a model with higher raw metrics (accuracy, precision, recall, F1, execution time) is an efficient requirements to claim this model outperform other models. Statistical significance testing helps to ensure the evaluation is sufficiently rigorous to conclude that higher performance does not happen by chance or by sampling. The table below describes how each five above evaluation metrics are statistically tested. 

**<u>Table 9: Evaluation Metrics and Statistical Significance using Cross Validation</u>** 

|<br>**Metric**|<br>**Definition**|<br>**How It is Compared (with**<br>**CV)**|<br>**Ensuring Robustness**|
|---|---|---|---|
|**Accuracy**|Ratio of correctly<br>classified cases<br>over total cases.|• Compute fold-level<br>accuracy in repeated<br>stratified k-fold CV<br>• Use the distribution of<br>scores across folds to<br>compare models.|Apply paired tests<br>(paired_t_-test or<br>Wilcoxon signed-rank)<br>or bootstrap 95%<br>confidence intervals to<br>assess statistical<br>significance (Marmion<br>_et al_., 2012).|
|**Precision**|Fraction of<br>predicted churners<br>that are actual<br>churners.|• Compute fold-level<br>precision in repeated<br>stratified k-fold CV<br>• Use the distribution of<br>scores across folds to<br>compare models.|Same significance logic<br>as accuracy: paired tests<br>or bootstrap 95% CIs<br>ensure results are not<br>due to chance.|
|**Recall**|Fraction of actual<br>churners correctly<br>identified<br>(especially<br>important in churn<br>contex.|• Compute fold-level recall<br>in repeated stratified k-<br>fold CV<br>• Use the distribution of<br>scores across folds to<br>compare models.|Same significance logic<br>as accuracy: paired tests<br>or bootstrap 95% CIs<br>validate recall<br>improvements across<br>models.|


### 7.8. Interpretability Analysis 

As described in objective section, beside quantitative evaluation metrics, model interpretability is another key component of the analysis phase. In the banking industry, the ability to explain predictions is critical not only for business stake holders such as compliance, risk management, and but also is crucial for gaining executives trust (Boozary, 2025). Well performance model, which fails to explain their predictions, are inappropriate for customer-facing decision systems deployment. Because those decision may directly result in poor customer retention strategies. 

This research will employ SHapley Additive exPlanations (SHAP) to provide a unified, modelagnostic approach to interpreting feature contributions. SHAP is built in cooperative game theory. Each feature contribution is computed as the average marginal contribution across all possible feature subsets. There are two main advantages of this approach. First, it generates overall explanations, which highlight features having most impact in churn prediction across entire dataset. Second, it also explains in detail why a specific customer is classified as likely to churn (AbdelAziz _et al_ ., 2025). 

For tree-based models (e.g., Random Forest, Gradient Boosted Trees, XGBoost), TreeSHAP will be used to optimize computational efficiency. For deep learning models, KernelSHAP will approximate Shapley values via sampling, ensuring interpretability remains feasible despite higher model complexity (Domingos, Ojeme & Daramola, 2021). 

The interpretability results will serve three purposes: 

- a. **Feature importance validation** involves cross-checking whether the highly weighted features correspond with domain expertise and existing banking literature. 

- b. **Model debugging** contains identifying whether specific features are introducing bias or creating false correlations. 

- c. **Actionable insights** : to guide retention strategies by revealing key behavioral or demographic drivers of churn. 

By integrating SHAP analysis into the evaluation pipeline, the study will bridge the gap between predictive accuracy and actionable decision-making, ensuring that model outputs are not only correct but also transparent and trustworthy. 

## **10.LIMITATIONS** 

This study acknowledges several methodological limitations, each of which has been considered carefully yet excluded from scope due to practical constraints. 

- a. Hyperparameter tuning for deep learning models 

While machine learning models (LR, DT, RF, XGBoost) will undergo systematic hyperparameter optimisation using Grid Search, Random Search, and Optuna, deep learning models (ANN, DNN) will not be fully tuned. This design decision rests on three considerations: 

- The dataset is relatively small (5,000 observations), which limits the marginal benefit of extensive hyperparameter search; overly complex tuning risks overfitting. 

- Full hyperparameter tuning for DL is computationally expensive, requiring hardware and time resources beyond the practical scope of this research. 

- Prior churn prediction studies provide empirically validated DL configurations that can be reasonably applied without exhaustive search (Amal _et al_ ., 2023; Ahmed _et al_ ., 2023; Zhang _et al_ ., 2024). 

Consequently, DL models are included as fair, literature-informed benchmarks rather than fully optimised candidates. 

- b. Scope of architectures 

Only baseline ANN and DNN are considered. More advanced deep learning architectures (e.g., CNNs, LSTMs, hybrid networks) are excluded, again due to computational constraints and dataset size limitations. Their exclusion narrows generalizability but ensures feasibility within a master’s-level research. 

- c. Dataset coverage 

The dataset is cross-sectional (5,000 customers, one row per customer) and lacks historical transactional sequences. As such, the study cannot evaluate time-series models or capture longitudinal churn behaviours. This limits the temporal dimension of churn prediction but reflects the constraints of the chosen data source. 

- d. Generalisability 

Results will be specific to the Kaggle dataset used. While the methodology is transferable, absolute performance levels may differ in proprietary banking datasets with richer features or larger sample sizes. 

These limitations are explicitly recognised as design trade-offs: the study prioritises methodological clarity, computational feasibility, and fairness over exhaustive model exploration. Future research should extend hyperparameter optimisation to deep learning architectures, 

incorporate temporal datasets, and evaluate a wider range of neural models under larger-scale conditions. 

## **REFERENCES** 

#### **Journal Articles** 

- Abbas, S.M., _et al._ (2023) Investigating customer churn in banking: A machine learning perspective. _Computers & Security_ , 123, 103041. Available at: <u>https://doi.org/10.1016/j.cose.2023.103041 (Accessed: 03 August 2025).</u> 

- AbdelAziz, A., Bekheet, K., Salah, A., El-Saber, H. and AbdelMoneim, M. (2025) A comprehensive evaluation of machine learning and deep learning models for churn prediction. _Information_ , 16(7), 537. Available at: <u>https://doi.org/10.3390/info16070537</u> (Accessed: 20 August 2025). 

- Abubakar, I., Ojo, J. and Bello, M. (2024) Improving churn detection in the banking sector: A machine learning approach with probability calibration techniques. _Electronics_ , 13(22), 4527. Available at: https://doi.org/10.3390/electronics13224527 (Accessed: 13 August 2025). 

- Adamu, H., _et al._ (2025) Customer churn prediction in banking sectors using a hyperparameter-tuned deep learning model. _Journal of Information Systems Engineering & Management_ , 10(31s), 5211. Available at: https://doi.org/10.52783/jisem.v10i31s.5211 (Accessed: 07 August 2025). 

- Ahmed, M., _et al._ (2023) Customer churn prediction using composite deep learning - 

- technique. _Scientific Reports_ , 13(1), 44396. Available at: <u>https://doi.org/10.1038/s41598 023-44396</u> (Accessed: 13 August 2025). 

- Adiputra, I.N.M., Wanchai, P. and Lin, P.-C. (2025) Optimized customer churn prediction using tabular generative adversarial network (GAN)-based hybrid sampling method and cost-sensitive learning. _PeerJ Computer Science_ , 11, e2949. Available at: <u>https://doi.org/10.7717/peerj-cs.2949</u> (Accessed: 14 August 2025). 

- Domingos, E., Ojeme, B. and Daramola, O. (2021) Experimental analysis of hyperparameters for deep learning-based churn prediction in the banking sector. _Computation_ , 9(3), 34. Available at: https://doi.org/10.3390/computation9030034 (Accessed: 04 August 2025). 

- Basit, A., Sheikh, S., Umer, M. and Syed, A. (2024) Comparative analysis of deep learning architectures for customer churn prediction in the banking sector. _Journal of Computers and Intelligent Systems_ . (In press). 

- Bhuria, R., _et al._ (2025) Ensemble-based customer churn prediction in banking: A voting classifier approach. _Sustainable Computing: Informatics and Systems_ . (In press). 

- Boozary, P. (2025) Enhancing customer retention with machine learning: Ensemble approaches. _Journal of Retail Analytics_ . (In press). 

- Eom, G. and Byeon, H. (2023) Searching for optimal oversampling to process imbalanced data: GAN and SMOTE. _Mathematics_ , 11(16), 3605. Available at: <u>https://doi.org/10.3390/math11163605 (Accessed: 04 August 2025).</u> 

- Li, X., _et al._ (2025a) Prediction of bank credit customer churn based on sampling and ML integration. _AIMS Mathematics_ . (In press). 

- Li, X., _et al._ (2025b) Prediction of bank credit customers churn based on ML and interpretability. _Data Science in Finance & Economics_ , 5(1), 1–20. Available at: <u>https://doi.org/10.3934/DSFE.2025002 (Accessed: 08 August 2025).</u> 

- Marmion, E.A., Parry, M.A., Stott, A.E.N. and Cornford, T.J.W. (2012) Comparing machine learning classifiers in potential distribution modelling. _Expert Systems with Applications_ , 39(8), 8083–8093. Available at: <u>https://doi.org/10.1016/j.eswa.2010.12.016</u> (Accessed: 08 August 2025). 

- Sezer, O.B., Gudelek, M.U. and Ozbayoglu, A.M. (2018) Deep learning with LSTM networks for financial market predictions. _European Journal of Operational Research_ , 270(2), 654–669. Available at: https://doi.org/10.1016/j.ejor.2017.11.054 (Accessed: 14 August 2025). 

- Sezer, O.B., Gudelek, M.U. and Ozbayoglu, A.M. (2020) Financial time series forecasting with deep learning: A systematic literature review. _Applied Soft Computing_ , 90, 106181. Available at: <u>https://doi.org/10.1016/j.asoc.2020.106181</u> (Accessed: 04 August 2025). 

- Singh, A., _et al._ (2024) Enhancing customer churn analysis by using real churn in banking with XAI. _International Journal of Advanced Computer Science and Applications_ , 16(5), 345–356. 

- Singh, H., _et al._ (2024) Investigating customer churn in banking: A machine learning comparative analysis. _Heliyon_ , 10(5), e15123. 

- Singh, P., _et al._ (2025) Ensemble-based customer churn prediction in banking: A voting ensemble approach. _Environmental Earth Sciences_ . Available at: <u>https://doi.org/10.1007/s43621-025-00807-8</u> (Accessed: 06 August 2025). 

- Singh, R., _et al._ (2023) Deep dive into churn prediction in the banking sector: The challenge of hyperparameter selection and imbalanced learning. _International Journal of Finance & Economics_ , 28(3), 2150–2172. Available at: https://doi.org/10.1002/ijfe.3078 (Accessed: 10 August 2025). 

- Tékouabou, F. (2022) Towards explainable machine learning for bank churn prediction: SMOTE + ensembles. _Mathematics_ , 10(14), 2379. Available at: <u>https://doi.org/10.3390/math10142379 (Accessed: 10 August 2025).</u> 

- Zhang, Y., _et al._ (2024) Customer churn prediction model based on hybrid neural networks (CCP-Net). _Scientific Reports_ , 14, 79603. Available at: <u>https://doi.org/10.1038/s41598-024-79603-9</u> (Accessed: 16 August 2025). 

#### **Books** 

- Molnar, C. (2020a) _Interpretable machine learning: A guide for making black box models explainable_ . 1st edn. Munich: Leanpub. 

- Molnar, C. (2020b) _Interpretable machine learning_ [online]. Available at: <u>https://christophm.github.io/interpretable-ml-book/ (Accessed: 02 August 2025).</u> 

#### **Working Papers** 

- Ma, T., van der Laan, E. and Reuvers, M. (2022) Interpretable machine learning for predicting customer churn in retail banking. Tilburg University Working Paper. 

#### **Conference Proceedings** 

- Murindanyi, D., Nagwovuma, A., _et al._ (2025) Explainable ensemble learning and trustworthy open AI for customer engagement prediction in retail banking. In: _Proceedings of the 2023 International Conference on Computing, Communication, and Intelligent Systems (IC3)_ . New York, NY: ACM. 
