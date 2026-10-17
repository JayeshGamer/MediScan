
# -*- coding: utf-8 -*-
"""Build a clean, IEEE-conference two-column DOCX for the MediScan audit paper."""
import os, sys, json
ROOT = r"D:\CapStone-2"
FIG = os.path.join(ROOT, "paper_figures")
RES = os.path.join(ROOT, "clean_study", "results")

REFS = [
 "W. Khan, N. Zaki, and L. Ali, \u201cIntelligent pneumonia identification from chest X-rays: A systematic literature review,\u201d IEEE Access, vol. 9, pp. 51\u202f747\u201351\u202f771, 2021.",
 "S. Solanki, U. P. Singh, S. S. Chouhan, and S. Jain, \u201cBrain tumor detection and classification using intelligence techniques: An overview,\u201d IEEE Access, vol. 11, pp. 12\u202f870\u201312\u202f886, 2023.",
 "J.-Y. Sun, S.-W. Lee, M.-C. Kang, S.-W. Kim, S.-Y. Kim, and S.-J. Ko, \u201cA novel gastric ulcer differentiation system using convolutional neural networks,\u201d Proc. IEEE 31st Int. Symp. Comput.-Based Med. Syst. (CBMS), pp. 351\u2013356, 2018.",
 "N. Noreen, S. Palaniappan, A. Qayyum, I. Ahmad, M. Imran, and M. Shoaib, \u201cA deep learning model based on concatenation approach for the diagnosis of brain tumor,\u201d IEEE Access, vol. 8, pp. 55\u202f135\u201355\u202f144, 2020.",
 "S. Budhiraja and N. Batra, \u201cDeep learning based approaches for gastric cancer detection and lesion segmentation in gastrointestinal endoscopy images,\u201d Proc. IEEE Int. Conf. Adv. Comput. Commun. Control, pp. 1\u20136, 2023.",
 "S. Sharma and K. Guleria, \u201cA deep learning based model for the detection of pneumonia from chest X-ray images using VGG-16 and neural networks,\u201d Procedia Comput. Sci., vol. 218, pp. 357\u2013366, 2023.",
 "P. Agrawal and S. Gandhi, \u201cPneumoSense++: An enhanced pneumonia detection with deep learning and comparative optimizer analysis,\u201d Procedia Comput. Sci., vol. 283, pp. 3407\u20133423, 2026.",
 "M. Ali et al., \u201cPneumonia detection using chest radiographs with novel EfficientNetV2L model,\u201d IEEE Access, vol. 12, pp. 24\u202f456\u201324\u202f467, 2024.",
 "S. Hatami, M. H. Olyaei, and R. Shamsaee, \u201cDetection and classification of gastric precancerous diseases using deep learning,\u201d Proc. 6th Iranian Conf. Signal Process. Intell. Syst. (ICSPIS), pp. 1\u20135, 2020.",
 "D. Li and S. Li, \u201cAn artificial intelligence deep learning platform achieves high diagnostic accuracy for Covid-19 pneumonia by reading chest X-ray images,\u201d iScience, vol. 25, p. 104031, 2022.",
 "J. Cheng, W. Yang, M. Huang, W. Huang, J. Jiang, Y. Zhou et al., \u201cRetrieval of brain tumors by adaptive spatial pooling and Fisher vector representation,\u201d PLOS ONE, vol. 11, no. 6, p. e0157112, 2016.",
 "M. Rizwan, A. Shabbir, A. R. Javed, M. Shabbir, T. Baker, and D. Al-Jumeily Obe, \u201cBrain tumor and glioma grade classification using Gaussian convolutional neural network,\u201d IEEE Access, vol. 10, pp. 29\u202f731\u201329\u202f740, 2022.",
 "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, \u201cGrad-CAM: Visual explanations from deep networks via gradient-based localization,\u201d in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618\u2013626.",
 "N. Shilpa, W. A. Banu, and P. B. Metre, \u201cRevolutionizing pneumonia diagnosis: AI-driven deep learning framework for automated detection from chest X-rays,\u201d IEEE Access, vol. 12, pp. 171\u202f601\u2013171\u202f616, 2024.",
 "D. Rickard, M. A. Kabir, and N. Homaira, \u201cMachine learning-based approaches for distinguishing viral and bacterial pneumonia in paediatrics: A scoping review,\u201d Comput. Methods Programs Biomed., vol. 258, p. 108512, 2025.",
 "K. Munir, M. U. Tanveer, H. J. Alyamani, A. Bermak, and A. U. Rehman, \u201cPneuX-Net: An enhanced feature extraction and transformation approach for pneumonia detection in X-ray images,\u201d IEEE Access, vol. 13, pp. 78\u202f912\u201378\u202f925, 2025.",
 "A. Kunjumon, S. V., and J. J. Nair, \u201cA hybrid deep learning framework for improved pneumonia detection in chest X-ray images,\u201d Procedia Comput. Sci., vol. 282, pp. 1043\u20131055, 2026.",
 "K. He, X. Zhang, S. Ren, and J. Sun, \u201cDeep residual learning for image recognition,\u201d in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770\u2013778.",
 "S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, \u201cCBAM: Convolutional block attention module,\u201d in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 3\u201319.",
 "Z. Jia and D. Chen, \u201cBrain tumor identification and classification of MRI images using deep learning techniques,\u201d IEEE Access, vol. 8, pp. 123\u202f783\u2013123\u202f797, 2020.",
 "D. S. Kermany, M. Goldbaum, W. Cai, C. C. Valentim, H. Liang, S. L. Baxter et al., \u201cIdentifying medical diagnoses and treatable diseases by image-based deep learning,\u201d Cell, vol. 172, no. 5, pp. 1122\u20131131, 2018.",
 "H. Olimboyev et al., \u201cDevelopment of the Gastro-AI model for diagnosing gastrointestinal diseases based on the HyperKvasir,\u201d Proc. IEEE Conf. Artif. Intell. Med., pp. 1\u20136, 2024.",
 "H. Borgli, V. Thambawita, P. Smedsrud, S. Hicks, D. Jha, S. L. Eskeland et al., \u201cHyperKvasir: A comprehensive multi-class image and video dataset for gastrointestinal endoscopy,\u201d Sci. Data, vol. 7, p. 283, 2020.",
 "S. Karim et al., \u201cDevelopments in brain tumor segmentation using MRI: Deep learning insights and future perspectives,\u201d IEEE Access, vol. 12, pp. 25\u202f958\u201325\u202f983, 2024.",
 "A. Younis et al., \u201cAbnormal brain tumors classification using ResNet50 and its comprehensive evaluation,\u201d IEEE Access, vol. 12, pp. 80\u202f470\u201380\u202f480, 2024.",
 "M. F. Almufareh, M. Imran, A. Khan, M. Humayun, and M. Asim, \u201cAutomated brain tumor segmentation and classification in MRI using YOLO-based deep learning,\u201d IEEE Access, vol. 12, pp. 16\u202f189\u201316\u202f204, 2024.",
 "S. Rajendran et al., \u201cAutomated segmentation of brain tumor MRI images using deep learning,\u201d IEEE Access, vol. 11, pp. 64\u202f758\u201364\u202f768, 2023.",
 "S. Asif, W. Yi, Q. U. Ain, J. Hou, T. Yi, and J. Si, \u201cImproving effectiveness of different deep transfer learning-based models for detecting brain tumors from MR images,\u201d IEEE Access, vol. 10, pp. 34\u202f716\u201334\u202f730, 2022.",
 "S. Ahmad and P. K. Choudhury, \u201cOn the performance of deep transfer learning networks for brain tumor detection using MR images,\u201d IEEE Access, vol. 10, pp. 59\u202f541\u201359\u202f556, 2022.",
 "M. U. Rehman et al., \u201cAn efficient deep learning model for brain tumour detection with privacy preservation,\u201d CAAI Trans. Intell. Technol., vol. 8, pp. 612\u2013624, 2023.",
 "S. S. Hussain, N. A. Wani, J. Kaur, N. Ahmad, and S. Ahmad, \u201cNext-generation automation in neuro-oncology: Advanced neural networks for MRI-based brain tumor segmentation and classification,\u201d IEEE Access, vol. 13, pp. 37\u202f483\u201337\u202f500, 2025.",
 "M. A. Ottom, H. A. Rahman, and I. D. Dinov, \u201cZnet: Deep learning approach for 2D MRI brain tumor segmentation,\u201d IEEE J. Transl. Eng. Health Med., vol. 10, pp. 1\u201310, 2022.",
 "R. Zaitoon and H. Syed, \u201cRU-Net2+: A deep learning algorithm for accurate brain tumor segmentation and survival rate prediction,\u201d IEEE Access, vol. 11, pp. 112\u202f626\u2013112\u202f643, 2023.",
 "R. Singh, S. Gupta, S. Bharany, A. Almogren, A. Altameem, and A. U. Rehman, \u201cEnsemble deep learning models for enhanced brain tumor classification by leveraging ResNet50 and EfficientNet-B7 on high-resolution MRI images,\u201d IEEE Access, vol. 12, pp. 171\u202f862\u2013171\u202f880, 2024.",
 "M. Tan and Q. V. Le, \u201cEfficientNet: Rethinking model scaling for convolutional neural networks,\u201d in Proc. Int. Conf. Mach. Learn. (ICML), 2019, pp. 6105\u20136114.",
 "Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, \u201cA ConvNet for the 2020s,\u201d in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 11\u202f976\u201311\u202f986.",
 "P. Szepesi and L. Szil\u00e1gyi, \u201cDetection of pneumonia using convolutional neural networks and deep learning,\u201d Biocybern. Biomed. Eng., vol. 42, pp. 1012\u20131022, 2022.",
 "S. Potharaju, S. N. Tambe, K. Dasari, N. Srikanth, R. Venkatarao, and S. Tambe, \u201cEnhanced X-ray image classification for pneumonia detection using deep learning based CBAM and SE mechanisms,\u201d Intell.-Based Med., vol. 11, p. 100142, 2025.",
 "H. Lamouadene et al., \u201cDetection of COVID-19, lung opacity, and viral pneumonia via X-ray using machine learning and deep learning,\u201d Comput. Biol. Med., vol. 170, p. 108012, 2025.",
 "S. Akshay, S. Kashyap, H. N. D. Patel, and Y. R. P. Kumar, \u201cHybrid deep learning and machine learning framework for automated pneumonia detection in chest X-ray images,\u201d MethodsX, vol. 15, p. 103729, 2025.",
 "M. F. Hashmi, A. A. Husain, D. M. Kotambkar, and K. D. Kumar, \u201cHybrid deep learning and handcrafted feature fusion for pneumonia detection in chest X-rays,\u201d J. Electr. Syst. Inf. Technol., vol. 12, p. 94, 2025.",
 "M. Fu, C. Tantithamthavorn, and T. Le, \u201cDA ViT: A domain-adapted vision transformer for automated pneumonia detection and explanation using chest X-ray images,\u201d IEEE Access, vol. 13, pp. 34\u202f120\u201334\u202f135, 2025.",
 "N. Rana and H. Marwaha, \u201cPneumonia detection from X-ray images using federated learning\u2014an unsupervised learning approach,\u201d Meas.: Sensors, vol. 35, p. 101189, 2025.",
 "M. R. Hasan, S. M. A. Ullah, and S. M. R. Islam, \u201cRecent advancement of deep learning techniques for pneumonia prediction from chest X-ray image,\u201d Medical Reports, vol. 7, p. 100106, 2024.",
 "M. Sun, K. Liang, W. Zhang, Q. Chang, and X. Zhou, \u201cNon-local attention and densely-connected convolutional neural networks for malignancy suspiciousness classification of gastric ulcer,\u201d IEEE Access, vol. 8, pp. 15\u202f812\u201315\u202f822, 2020.",
 "S.-A. Lee, H. C. Cho, and H.-C. Cho, \u201cA novel approach for increased convolutional neural network performance in gastric-cancer classification using endoscopic images,\u201d IEEE Access, vol. 9, pp. 51\u202f847\u201351\u202f854, 2021.",
 "T. A. Kuchkorov, N. Q. Sabitova, and T. D. Ochilov, \u201cDetection of gastric ulcers and lesions applying CNN architecture,\u201d Proc. Int. Sci. Tech. Conf. Comput. Sci. Eng. Technol., pp. 200\u2013205, 2022.",
 "S. Ahmad, J.-S. Kim, D. K. Park, and T. Whangbo, \u201cAutomated detection of gastric lesions in endoscopic images by leveraging attention-based YOLOv7,\u201d IEEE Access, vol. 11, pp. 73\u202f418\u201373\u202f429, 2023.",
 "Y. Tan et al., \u201cDeep learning-based AI system for real-time differentiation of benign and malignant gastric ulcers,\u201d BMC Gastroenterol., vol. 26, p. 378, 2026.",
 "B. S. Kim et al., \u201cEnhanced multi-class pathology lesion detection in gastric neoplasms using deep learning-based approach and validation,\u201d Sci. Rep., vol. 14, p. 11527, 2024.",
 "X. Zheng, Z. Zeng, C. Ma, Q. Chang, Z. Zhao, and X. Yang, \u201cMalignancy suspicious region guided deep neural networks for gastric ulcer classification,\u201d Proc. IEEE Int. Conf. Robot. Autom. (ICRA), pp. 1\u20136, 2021.",
 "M. Ramamoorthy et al., \u201cEarly cancer diagnosis and ulcer classifications in endoscopy videos using deep learning approach,\u201d Proc. IEEE Int. Conf. Comput. Intell. Computing Res., pp. 1\u20136, 2023.",
 "G. P. Lee, Y. J. Kim, D. K. Park, Y. J. Kim, S. K. Han, and K. G. Kim, \u201cGastro-BaseNet: A specialized pre-trained model for enhanced gastroscopic data classification and diagnosis of gastric cancer and ulcer,\u201d Diagnostics, vol. 14, p. 75, 2024.",
 "R. Yang et al., \u201cArtificial intelligence efficiently predicts gastric lesions, Helicobacter pylori infection and lymph node metastasis upon endoscopic images,\u201d Chin. J. Cancer Res., vol. 36, pp. 489\u2013502, 2024.",
 "M. Zubair, M. Owais, T. Mahmood, S. Iqbal, S. M. Usman, and I. Hussain, \u201cEnhanced gastric cancer classification and quantification interpretable framework using digital histopathology images,\u201d Sci. Rep., vol. 14, p. 8912, 2024.",
 "A. Saba, J. Amin, and M. U. Ali, \u201cDeep Q-learning for gastrointestinal disease detection and classification,\u201d Bioengineering, vol. 12, p. 1184, 2025.",
 "Y. Masmoudi, M. Ramzan, S. A. Khan, and M. Habib, \u201cOptimal feature extraction and ulcer classification from WCE image data using deep learning,\u201d Soft Comput., vol. 26, pp. 7571\u20137583, 2022.",
 "S. Gnanapriya, D. Reshma, and M. Varun, \u201cIdentification of gastric abnormalities with synchronous biopsy site localization from endoscope imaging in gastro-intestinal tract using deep learning technique,\u201d Proc. IEEE Int. Conf. Circuit Power Comput. Technol., pp. 1\u20136, 2023.",
 "H. J. Yoon and J.-H. Kim, \u201cLesion-based convolutional neural network in diagnosis of early gastric cancer,\u201d Clin. Endosc., vol. 53, pp. 127\u2013131, 2020.",
 "RSNA, \u201cRSNA pneumonia detection challenge,\u201d Radiological Society of North America, 2018. [Online]. Available: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge",
 "B. H. Menze et al., \u201cThe multimodal brain tumor image segmentation benchmark (BRATS),\u201d IEEE Trans. Med. Imag., vol. 34, no. 10, pp. 1993\u20132024, 2015.",
 "P. H. Smedsrud et al., \u201cKvasir-Capsule, a video capsule endoscopy dataset,\u201d Sci. Data, vol. 8, p. 142, 2021.",
]

def load_json(name):
    p = os.path.join(RES, name)
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None
