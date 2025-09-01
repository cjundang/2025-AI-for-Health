# Week 11 NLP and Text Analytics

## **1. Introduction to NLP and Text Analytics**

### **1.1 ความหมายของ NLP (Natural Language Processing)**

Natural Language Processing (NLP) คือ สาขาหนึ่งของ **ปัญญาประดิษฐ์ (AI)** และ **วิทยาการข้อมูล (Data Science)** ที่เกี่ยวข้องกับการทำให้คอมพิวเตอร์สามารถ:

* **เข้าใจ (Understand)** ความหมายของภาษามนุษย์
* **วิเคราะห์ (Analyze)** เนื้อหาและบริบทในข้อความ
* **สร้าง (Generate)** ข้อความใหม่ที่มีความหมายสอดคล้องกับภาษามนุษย์

NLP รวมองค์ความรู้จาก **ภาษาศาสตร์ (Linguistics)**, **คณิตศาสตร์**, **สถิติ**, และ **Machine Learning** เพื่อให้คอมพิวเตอร์ประมวลผลข้อมูลในรูปแบบข้อความ (Text) และเสียง (Speech) ได้อย่างมีประสิทธิภาพ



### **1.2 ความหมายของ Text Analytics**

Text Analytics เป็นกระบวนการ **วิเคราะห์เนื้อหาข้อความ** เพื่อค้นหาความหมายเชิงลึก (Insight) จากข้อมูลจำนวนมาก (Unstructured Data)
โดยใช้เทคนิคต่าง ๆ เช่น:

* การนับคำ (Word Frequency)
* การวิเคราะห์ความรู้สึก (Sentiment Analysis)
* การจัดกลุ่มหัวข้อ (Topic Modeling)
* การจำแนกข้อความ (Text Classification)



### **1.3 ความแตกต่างระหว่าง NLP, Text Mining และ Text Analytics**

| **คุณสมบัติ**  | **NLP**                            | **Text Mining**                | **Text Analytics**                  |
| -------------- | ---------------------------------- | ------------------------------ | ----------------------------------- |
| **จุดประสงค์** | ทำให้คอมพิวเตอร์เข้าใจภาษามนุษย์   | สกัดข้อมูลที่ซ่อนอยู่ในข้อความ | วิเคราะห์ข้อความเพื่อสร้าง Insight  |
| **เทคนิคหลัก** | Machine Learning, Deep Learning    | Keyword Extraction, Clustering | Sentiment, Topic Modeling           |
| **ผลลัพธ์**    | การแปลภาษา, Chatbot, Summarization | Pattern Discovery              | Business Insights, Decision Support |



### **1.4 ความสำคัญของ NLP และ Text Analytics**

* ช่วยองค์กร **วิเคราะห์ข้อมูลขนาดใหญ่** ที่มาจากข้อความ เช่น Social Media, Email, Chat Logs
* เพิ่มความสามารถของ **Chatbots และ Virtual Assistants**
* สนับสนุน **Business Intelligence** และการตัดสินใจเชิงกลยุทธ์
* ประยุกต์ใช้ในหลายอุตสาหกรรม เช่น การตลาด, การแพทย์, ความปลอดภัยไซเบอร์ และระบบแนะนำเนื้อหา (Recommendation Systems)

## **2. Text Preprocessing**

Text Preprocessing คือกระบวนการเตรียมข้อความ (Text Data) ให้อยู่ในรูปแบบที่เหมาะสมสำหรับการวิเคราะห์และสร้างโมเดล Machine Learning หรือ NLP โดยมีขั้นตอนสำคัญหลายขั้นตอน ดังนี้:


### **2.1 Tokenization (การตัดคำหรือแบ่งข้อความ)**

**ความหมาย:**
กระบวนการแบ่งข้อความขนาดใหญ่ (Document หรือ Sentence) ออกเป็นหน่วยย่อยที่เรียกว่า **Tokens** ซึ่งอาจเป็นคำ (Word Tokenization) หรือประโยค (Sentence Tokenization)

**ตัวอย่าง:**

```
ข้อความ: "Natural Language Processing is fun"
Tokens: ["Natural", "Language", "Processing", "is", "fun"]
```

**เทคนิคที่ใช้บ่อย:**

* **Whitespace Tokenization** → แบ่งโดยใช้ช่องว่าง
* **Regex Tokenization** → ใช้ Regular Expressions
* **Tokenizer Libraries** → เช่น **NLTK**, **spaCy**, **KoNLPy** (สำหรับภาษาไทย)


### **2.2 Stopword Removal (การลบคำฟุ่มเฟือย)**

**ความหมาย:**
Stopwords คือคำที่มีความถี่สูงแต่มีความหมายเชิงวิเคราะห์ต่ำ เช่น **"is", "the", "and"** ในภาษาอังกฤษ หรือ **"คือ", "ว่า", "และ"** ในภาษาไทย

**เหตุผลที่ต้องลบ:**

* ลดขนาดของชุดข้อมูล
* เพิ่มประสิทธิภาพในการวิเคราะห์ข้อความ
* ทำให้โมเดลโฟกัสที่คำสำคัญจริง ๆ

**วิธีการลบ:**
ใช้ **Stopword Dictionary** จาก **NLTK, spaCy, PyThaiNLP** หรือสร้างลิสต์เองตามบริบทงาน



### **2.3 Stemming และ Lemmatization**

#### **(1) Stemming**

* **ความหมาย:** ตัดคำให้อยู่ในรากศัพท์ โดยไม่สนใจความถูกต้องทางไวยากรณ์
* **เครื่องมือยอดนิยม:** **Porter Stemmer**, **Snowball Stemmer**
* **ตัวอย่าง:**

  ```
  "running", "runner", "ran" → "run"
  ```

#### **(2) Lemmatization**

* **ความหมาย:** คล้ายกับ Stemming แต่จะใช้พจนานุกรมและกฎทางภาษาศาสตร์เพื่อให้ได้คำรากที่ถูกต้องจริง ๆ
* **เครื่องมือยอดนิยม:** **WordNet Lemmatizer**
* **ตัวอย่าง:**

  ```
  "better" → "good"
  "studies" → "study"
  ```

> **ข้อแตกต่างหลัก:** Lemmatization ให้ผลลัพธ์ที่แม่นยำกว่า แต่ใช้เวลาและทรัพยากรมากกว่า Stemming


### **2.4 Lowercasing และ Normalization**

* **Lowercasing** → แปลงตัวอักษรทั้งหมดให้เป็นตัวพิมพ์เล็ก เช่น

  ```
  "Natural" → "natural"
  ```
* **Normalization** → ทำให้ข้อความอยู่ในรูปแบบที่สอดคล้องกัน เช่น:

  * ลบอักขระพิเศษ เช่น `#`, `@`, `!`
  * แปลงตัวย่อ เช่น `u` → `you`
  * จัดการตัวสะกดผิดหรือเว้นวรรคผิด



### **2.5 Part-of-Speech (POS) Tagging**

**ความหมาย:**
การกำหนดชนิดของคำ เช่น **Noun (คำนาม)**, **Verb (คำกริยา)**, **Adjective (คำคุณศัพท์)** ให้กับแต่ละ Token

**ตัวอย่าง:**

```
ประโยค: "ChatGPT generates text"
ผลลัพธ์: [("ChatGPT", NNP), ("generates", VBZ), ("text", NN)]
```

**ประโยชน์:**

* ช่วยในการสร้างโมเดล **Named Entity Recognition (NER)**
* สนับสนุนการทำ **Dependency Parsing** และ **Semantic Analysis**



### **2.6 Text Cleaning เพิ่มเติม**

* ลบ **HTML Tags** และ **URLs**
* ลบ **ตัวเลข** หากไม่จำเป็น
* ลบ **Emojis** หรือ **สัญลักษณ์พิเศษ**
* การทำ **Spell Correction** เช่น `goood` → `good`



### **สรุป**

Text Preprocessing เป็นขั้นตอนสำคัญใน NLP และ Text Analytics เพราะช่วย:

* ลดสัญญาณรบกวนในข้อมูล
* ทำให้โมเดล Machine Learning และ Deep Learning มีความแม่นยำสูงขึ้น
* เตรียมข้อมูลให้อยู่ในรูปแบบที่ง่ายต่อการนำไป **Feature Extraction** และ **Modeling**



## **3. Feature Extraction**

Feature Extraction คือกระบวนการแปลงข้อความ (Text Data) ให้อยู่ในรูปแบบตัวเลขหรือ **เวกเตอร์ (Vectors)** เพื่อให้โมเดล Machine Learning หรือ NLP สามารถนำไปวิเคราะห์ได้
โดยทั่วไป เทคนิคที่ใช้บ่อย ได้แก่ **Bag of Words**, **TF**, **IDF**, **TF-IDF**, และ **n-Grams** ซึ่งผมจะอธิบายรายละเอียดแต่ละเทคนิคด้านล่าง



### **3.1 Bag of Words (BoW)**

**ความหมาย:**
เป็นเทคนิคพื้นฐานในการแทนข้อความเป็น **เวกเตอร์ของจำนวนคำ** โดยไม่สนใจลำดับคำในประโยค
แนวคิดคือสร้าง **Vocabulary (คลังคำ)** จากข้อความทั้งหมด และแทนเอกสารแต่ละฉบับเป็นจำนวนครั้งที่คำปรากฏในเอกสารนั้น

**ขั้นตอน:**

1. รวบรวมข้อความทั้งหมด
2. สร้าง **คลังคำ (Vocabulary)**
3. นับจำนวนคำในแต่ละเอกสาร

**ตัวอย่าง:**

```
ข้อความ: ["I love NLP", "I love AI"]
Vocabulary: ["I", "love", "NLP", "AI"]
BoW Representation:
Doc1 → [1, 1, 1, 0]
Doc2 → [1, 1, 0, 1]
```

**ข้อดี:**

* เข้าใจง่าย และใช้เวลาคำนวณน้อย
* ทำงานได้ดีกับข้อความสั้น ๆ

**ข้อเสีย:**

* ไม่คำนึงถึงลำดับคำ (**Orderless**)
* หากข้อมูลมีขนาดใหญ่ จะเกิด **Sparse Matrix** ซึ่งใช้หน่วยความจำมาก


### **3.2 Term Frequency (TF)**

**ความหมาย:**
TF คือ **ความถี่ของคำ** ที่ปรากฏในเอกสารหนึ่ง ๆ โดยคิดเป็นสัดส่วนของจำนวนครั้งที่คำปรากฏเทียบกับจำนวนคำทั้งหมดในเอกสาร

**สูตรคำนวณ:**

$$
TF(t,d) = \frac{\text{จำนวนครั้งที่คำ t ปรากฏในเอกสาร d}}{\text{จำนวนคำทั้งหมดในเอกสาร d}}
$$

**ตัวอย่าง:**
ถ้าในเอกสาร A มี 100 คำ และคำว่า “data” ปรากฏ 5 ครั้ง

$$
TF(\text{data}, A) = \frac{5}{100} = 0.05
$$

**ประโยชน์:**

* เหมาะสำหรับการวิเคราะห์ความถี่คำภายในเอกสารเดียวกัน


### **3.3 Inverse Document Frequency (IDF)**

**ความหมาย:**
IDF ใช้ในการลดความสำคัญของคำที่พบบ่อยเกินไปในทุกเอกสาร เช่น **“the”, “is”, “and”**
ถ้าคำปรากฏในทุกเอกสาร จะได้ค่า IDF ต่ำ แต่ถ้าคำพบเฉพาะในบางเอกสาร ค่า IDF จะสูง

**สูตรคำนวณ:**

$$
IDF(t) = \log \left(\frac{N}{1 + DF(t)}\right)
$$

โดยที่:

* **N** = จำนวนเอกสารทั้งหมด
* **DF(t)** = จำนวนเอกสารที่มีคำ **t**


### **3.4 TF-IDF Weighting**

**ความหมาย:**
TF-IDF คือการรวมแนวคิดของ **TF** และ **IDF** เพื่อสร้างค่าความสำคัญของคำที่สมดุลมากขึ้น

* ถ้าคำปรากฏบ่อยในเอกสาร → **TF สูง**
* ถ้าคำพบในทุกเอกสาร → **IDF ต่ำ** → ลดความสำคัญของคำ
* คำที่สำคัญที่สุดจะมี **TF สูง** และ **IDF สูง**

**สูตรคำนวณ:**

$$
TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)
$$

**ตัวอย่างการประยุกต์ใช้:**

* การจัดลำดับเอกสารใน **Search Engines**
* การทำ **Keyword Extraction**
* การสร้างโมเดลสำหรับ **Text Classification**


### **3.5 n-Grams**

**ความหมาย:**
n-Grams คือการจับกลุ่มคำที่อยู่ติดกันจำนวน **n คำ** เพื่อรักษาลำดับและบริบทของคำในข้อความ

**ประเภทของ n-Grams:**

* **Unigram (n=1):** `"machine"`, `"learning"`
* **Bigram (n=2):** `"machine learning"`, `"deep learning"`
* **Trigram (n=3):** `"natural language processing"`

**ข้อดี:**

* เก็บข้อมูลบริบทของคำได้ดีกว่า BoW
* ใช้กับงาน **Text Generation**, **Machine Translation**, **Chatbots**

**ข้อเสีย:**

* ถ้า **n** มีค่าสูงเกินไป → ขนาด **Feature Space** จะใหญ่
* ต้องการข้อมูลจำนวนมากเพื่อเทรนโมเดล


### **3.6 การเลือกวิธี Feature Extraction**

| **เทคนิค** | **จุดเด่น**                   | **จุดด้อย**                 | **เหมาะกับ**              |
| ---------- | ----------------------------- | --------------------------- | ------------------------- |
| BoW        | เข้าใจง่าย คำนวณเร็ว          | ไม่สนใจลำดับคำ              | ข้อความสั้น ๆ             |
| TF         | วัดความถี่คำในเอกสารเดียว     | ไม่พิจารณาบริบท             | การวิเคราะห์คำเฉพาะเอกสาร |
| IDF        | ลดน้ำหนักคำที่พบบ่อยมากเกินไป | อาจไม่เพียงพอถ้าใช้เดี่ยว   | การกรอง Stopwords         |
| TF-IDF     | สมดุลระหว่าง TF และ IDF       | ยังไม่เก็บความสัมพันธ์ของคำ | Text Classification       |
| n-Grams    | เก็บบริบทได้ดีกว่า BoW        | ใช้พื้นที่หน่วยความจำสูง    | Machine Translation, NLP  |


### **สรุป**

* **Feature Extraction** เป็นขั้นตอนสำคัญในการแปลงข้อความให้โมเดลเข้าใจ
* เทคนิคยอดนิยม ได้แก่ **BoW, TF, IDF, TF-IDF และ n-Grams**
* ในงาน **Text Classification**, **Sentiment Analysis**, และ **Information Retrieval** มักใช้ **TF-IDF**
* ถ้างานต้องการบริบทของคำ → ใช้ **n-Grams** ร่วมกับเทคนิคอื่น ๆ

## **4. Sentiment Analysis**

### **4.1 ความหมายของ Sentiment Analysis**

Sentiment Analysis หรือ **การวิเคราะห์ความคิดเห็น/ความรู้สึก** คือกระบวนการวิเคราะห์ข้อความ (Text Data) เพื่อระบุอารมณ์ ความรู้สึก หรือทัศนคติของผู้เขียน/ผู้พูดว่าเป็น:

* **เชิงบวก (Positive)**
* **เชิงลบ (Negative)**
* **เป็นกลาง (Neutral)**

เทคนิคนี้เป็นส่วนหนึ่งของ **Natural Language Processing (NLP)** และถูกใช้กันอย่างแพร่หลายในงาน **Text Analytics** และ **Opinion Mining**



### **4.2 ประเภทของ Sentiment Analysis**

Sentiment Analysis แบ่งออกเป็นหลายระดับตามความละเอียดที่ต้องการ:

#### **(1) Document-Level Sentiment Analysis**

* วิเคราะห์ความคิดเห็นโดยพิจารณาจากทั้งเอกสาร
* ใช้ในรีวิวหนังสือ บทความ หรือโพสต์ขนาดยาว
* ตัวอย่าง: บทวิจารณ์ภาพยนตร์ทั้งเรื่อง → “ภาพยนตร์เรื่องนี้ยอดเยี่ยมมาก” → Positive

#### **(2) Sentence-Level Sentiment Analysis**

* วิเคราะห์ในระดับประโยค
* เหมาะกับข้อความสั้น ๆ เช่น **ทวีต** หรือ **คอมเมนต์**
* ตัวอย่าง: “I love this phone, but the battery life is terrible”

  * ประโยคแรก: Positive
  * ประโยคหลัง: Negative

#### **(3) Aspect-Based Sentiment Analysis (ABSA)**

* วิเคราะห์ความคิดเห็นในมุมมองแง่มุม (Aspect) ของสินค้า/บริการ
* ตัวอย่าง: “The camera is amazing but the battery sucks”

  * Aspect “Camera” → Positive
  * Aspect “Battery” → Negative



### **4.3 วิธีการทำ Sentiment Analysis**

โดยทั่วไป มี 3 แนวทางหลักที่ใช้กัน:

#### **(1) Rule-Based Approach**

* ใช้ **Lexicon** หรือชุดคำศัพท์ที่บ่งบอกอารมณ์ เช่น

  * Positive words → “good”, “great”, “fantastic”
  * Negative words → “bad”, “terrible”, “awful”
* นับจำนวนคำเชิงบวกและลบ → สรุปว่าเป็น **Positive** หรือ **Negative**

**ข้อดี:** ทำงานง่าย ไม่ต้องใช้ข้อมูลจำนวนมาก
**ข้อเสีย:** ความแม่นยำต่ำ ไม่เข้าใจบริบทซับซ้อน



#### **(2) Machine Learning-Based Approach**

* ใช้ **โมเดลการเรียนรู้ของเครื่อง (ML Models)** เช่น:

  * **Naïve Bayes**
  * **Support Vector Machines (SVM)**
  * **Logistic Regression**
* ต้องมี **ข้อมูลที่ติดป้ายกำกับ (Labeled Data)** เช่น:

  ```
  ข้อความ: "This product is amazing" → Label: Positive
  ```
* ขั้นตอนการทำงาน:

  1. **Text Preprocessing** → Tokenization, Stopword Removal, Lemmatization
  2. **Feature Extraction** → TF-IDF, n-Grams, Word Embeddings
  3. **Model Training** → สร้างโมเดลจากข้อมูลที่มี
  4. **Prediction** → ทำนายความรู้สึกของข้อความใหม่

**ข้อดี:** แม่นยำสูงเมื่อมีข้อมูลคุณภาพ
**ข้อเสีย:** ต้องใช้ข้อมูลขนาดใหญ่และเวลาฝึกสอนโมเดล



#### **(3) Deep Learning-Based Approach**

* ใช้ **Neural Networks** และ **Transformer Models** ที่ทันสมัย เช่น:

  * **RNN / LSTM / GRU** → ใช้กับลำดับข้อความ
  * **BERT, RoBERTa, GPT** → ใช้ Contextual Embeddings
* สามารถเรียนรู้บริบทของคำและโครงสร้างภาษาที่ซับซ้อนได้ดีกว่าโมเดลแบบดั้งเดิม

**ข้อดี:** ความแม่นยำสูงมาก เหมาะกับงานที่ซับซ้อน
**ข้อเสีย:** ต้องใช้ทรัพยากรสูงและข้อมูลปริมาณมาก



### **4.4 ตัวอย่างการทำงานของ Sentiment Analysis**

**ข้อความ:**

> “I really love this phone. The screen is beautiful, but the battery drains too fast.”

**ผลลัพธ์:**

* Document-Level → **Neutral** (มีทั้งบวกและลบ)
* Sentence-Level →

  * “I really love this phone.” → Positive
  * “The screen is beautiful.” → Positive
  * “The battery drains too fast.” → Negative
* Aspect-Based →

  * Screen → Positive
  * Battery → Negative



### **4.5 การประเมินผล (Evaluation Metrics)**

โมเดล Sentiment Analysis มักใช้ตัวชี้วัดดังนี้:

* **Accuracy** → ความถูกต้องทั้งหมด
* **Precision** → สัดส่วนของข้อความที่ทำนายว่า Positive และถูกจริง
* **Recall** → ความสามารถในการจับข้อความ Positive ทั้งหมดได้ครบหรือไม่
* **F1-Score** → ค่าที่สมดุลระหว่าง Precision และ Recall

 
### **4.6 การประยุกต์ใช้งาน Sentiment Analysis**

1. **Social Media Analytics** → วิเคราะห์ความคิดเห็นจาก Twitter, Facebook, TikTok
2. **Customer Feedback Analysis** → วิเคราะห์รีวิวสินค้า/บริการ
3. **Brand Monitoring** → ประเมินภาพลักษณ์แบรนด์
4. **Financial Market Prediction** → วิเคราะห์ข่าวและทวีตที่กระทบต่อหุ้น
5. **Chatbots และ Virtual Assistants** → ทำให้ระบบตอบสนองอารมณ์ผู้ใช้ได้ดีขึ้น

 
### **สรุป**

* Sentiment Analysis คือกระบวนการวิเคราะห์ข้อความเพื่อระบุ **อารมณ์/ทัศนคติ** ของผู้เขียน
* มี 3 แนวทางหลัก: **Rule-Based**, **Machine Learning-Based**, และ **Deep Learning-Based**
* งานประยุกต์ครอบคลุมหลากหลาย เช่น Social Media Analytics, Customer Feedback, และ Financial Forecasting


## **5. Topic Modeling**

### **5.1 ความหมายของ Topic Modeling**

Topic Modeling คือเทคนิคใน **Natural Language Processing (NLP)** และ **Text Analytics** ที่ใช้สำหรับ:

* ค้นหา **หัวข้อซ่อนเร้น (Hidden Topics)** ในชุดข้อมูลข้อความจำนวนมาก
* จัดกลุ่มเอกสารตามเนื้อหา โดยไม่ต้องมี **Label** (เป็น **Unsupervised Learning**)
* ช่วยให้เราเข้าใจข้อมูลเชิงลึก (Insight) จากเอกสารที่ไม่ได้มีโครงสร้าง (Unstructured Data)

**ตัวอย่าง:**
สมมติว่าเรามี **รีวิวสินค้า 10,000 รายการ** → Topic Modeling จะช่วยบอกว่ารีวิวเหล่านี้พูดถึงหัวข้อหลัก ๆ อะไรบ้าง เช่น:

* คุณภาพสินค้า
* ราคา
* บริการหลังการขาย


### **5.2 วิธีการทำงานของ Topic Modeling**

กระบวนการทำงานหลักมี 4 ขั้นตอนสำคัญ:

1. **Text Preprocessing**

   * Tokenization, Stopword Removal, Stemming/Lemmatization
   * สร้าง **Document-Term Matrix (DTM)** หรือ **TF-IDF Matrix**

2. **เลือกอัลกอริทึมสำหรับ Topic Modeling**

   * **Latent Dirichlet Allocation (LDA)**
   * **Non-negative Matrix Factorization (NMF)**
   * **Latent Semantic Analysis (LSA)**

3. **กำหนดจำนวนหัวข้อ (Number of Topics, K)**

   * กำหนดจำนวนหัวข้อที่ต้องการให้โมเดลค้นหา
   * ตัวอย่าง: ถ้าต้องการให้รีวิวสินค้ามี 3 หัวข้อหลัก → ตั้ง K = 3

4. **วิเคราะห์ผลลัพธ์**

   * โมเดลจะคืนรายการ **Top Words** ของแต่ละหัวข้อ
   * นำคำเหล่านี้ไปตีความเพื่อระบุหัวข้อของกลุ่มเอกสาร



### **5.3 เทคนิคยอดนิยมสำหรับ Topic Modeling**

#### **(1) Latent Dirichlet Allocation (LDA)**

* เป็นเทคนิคที่นิยมใช้มากที่สุด
* สมมติว่า:

  * เอกสารหนึ่ง ๆ ประกอบด้วย **หลายหัวข้อ**
  * แต่ละหัวข้อถูกแทนด้วย **การกระจายของคำ (Word Distribution)**
* LDA ใช้ **Bayesian Inference** เพื่อคำนวณความน่าจะเป็นที่เอกสารหนึ่งเกี่ยวข้องกับแต่ละหัวข้อ

**ตัวอย่างผลลัพธ์:**

```
Topic 1 → ["camera", "lens", "photo", "image"]
Topic 2 → ["battery", "charging", "power", "life"]
Topic 3 → ["price", "value", "cost", "cheap"]
```

→ เราตีความได้ว่า:

* **Topic 1:** คุณภาพกล้อง
* **Topic 2:** แบตเตอรี่
* **Topic 3:** ราคา



#### **(2) Non-negative Matrix Factorization (NMF)**

* ใช้หลักการ **แยกเมทริกซ์ (Matrix Factorization)** โดยให้ค่าทุกตัวเป็น **บวก (Non-negative)**
* ทำงานได้ดีเมื่อใช้ **TF-IDF Matrix**
* มีความแม่นยำสูงกว่า LDA เมื่อข้อมูลมีขนาดไม่ใหญ่มาก

**ข้อดีของ NMF:**

* ทำงานเร็ว
* ง่ายต่อการตีความผลลัพธ์
* เหมาะกับข้อมูลที่มีมิติสูง (High-dimensional Data)


#### **(3) Latent Semantic Analysis (LSA)**

* ใช้ **Singular Value Decomposition (SVD)** ในการลดมิติของข้อมูล
* ค้นหาความสัมพันธ์เชิงซ่อนเร้นระหว่าง **คำ (Terms)** และ **เอกสาร (Documents)**

**ข้อเสีย:**

* ผลลัพธ์ไม่สามารถตีความได้ง่ายเหมือน LDA
* ไม่เหมาะกับข้อความที่ซับซ้อน


### **5.4 การประเมินคุณภาพของ Topic Modeling**

การเลือกจำนวนหัวข้อ (**K**) ที่เหมาะสมและประเมินคุณภาพของโมเดลเป็นสิ่งสำคัญ
มีวิธีการประเมินที่นิยมใช้ ได้แก่:

1. **Coherence Score**

   * ใช้วัดความสัมพันธ์ของคำในหัวข้อ
   * ค่าใกล้ 1 → โมเดลจับหัวข้อได้ดี

2. **Perplexity Score**

   * วัดความสามารถของโมเดลในการทำนายคำที่หายไป
   * ค่า Perplexity ต่ำ → โมเดลดีกว่า

3. **Human Evaluation**

   * ให้ผู้เชี่ยวชาญช่วยตรวจสอบว่าหัวข้อที่โมเดลสร้างมามีความสมเหตุสมผลหรือไม่

### **5.5 ตัวอย่างการประยุกต์ใช้ Topic Modeling**

1. **การวิเคราะห์รีวิวสินค้า (Product Review Analysis)**

   * ใช้ค้นหาว่าลูกค้าพูดถึงอะไรบ่อย เช่น **ราคา**, **คุณภาพ**, **การบริการ**
2. **Social Media Analytics**

   * วิเคราะห์โพสต์ Twitter, Facebook เพื่อดูประเด็นร้อน
3. **Customer Feedback**

   * ใช้สรุปข้อร้องเรียนของลูกค้าและแนวโน้มความพึงพอใจ
4. **Research Paper Clustering**

   * จัดกลุ่มบทความวิจัยตามหัวข้ออัตโนมัติ
5. **ข่าวและบทความออนไลน์**

   * จัดหมวดหมู่ข่าว เช่น **การเมือง**, **กีฬา**, **เศรษฐกิจ**



### **5.6 เปรียบเทียบ LDA, NMF และ LSA**

| **คุณสมบัติ**    | **LDA**        | **NMF**              | **LSA**         |
| ---------------- | -------------- | -------------------- | --------------- |
| ประเภทโมเดล      | Probabilistic  | Matrix Factorization | SVD-based       |
| ข้อมูลอินพุต     | BoW / TF-IDF   | TF-IDF               | BoW / TF-IDF    |
| ความเร็ว         | ปานกลาง        | เร็ว                 | เร็วมาก         |
| การตีความผลลัพธ์ | ง่าย           | ง่าย                 | ค่อนข้างยาก     |
| เหมาะกับ         | ข้อมูลจำนวนมาก | ข้อมูลขนาดกลาง       | การลดมิติข้อมูล |

### **สรุป**

* **Topic Modeling** เป็นเทคนิคสำคัญในการจัดกลุ่มเอกสารโดยอัตโนมัติ
* เทคนิคที่นิยมใช้: **LDA**, **NMF**, และ **LSA**
* ใช้ประโยชน์ได้ในหลากหลายด้าน เช่น **Social Media Analytics**, **Customer Feedback**, **Product Review Analysis** และ **Research Paper Categorization**
* ถ้าต้องการตีความง่าย → ใช้ **LDA**
* ถ้าต้องการความเร็วและแม่นยำ → ใช้ **NMF**



## **6. Named Entity Recognition (NER)**

### **6.1 ความหมายของ NER**

**Named Entity Recognition (NER)** คือกระบวนการใน **Natural Language Processing (NLP)** ที่ใช้สำหรับ:

* **ตรวจจับ (Detection)** และ **ระบุชื่อเฉพาะ (Recognition)** ในข้อความ
* ทำการ **จำแนกประเภทของเอนทิตี (Entities)** เช่น:

  * **PERSON** → ชื่อบุคคล เช่น *“Elon Musk”*
  * **ORG** → ชื่อองค์กร เช่น *“Google”*
  * **GPE** → ชื่อประเทศ เมือง หรือสถานที่ เช่น *“Thailand”*
  * **DATE / TIME** → วัน เวลา
  * **MONEY / PERCENT** → จำนวนเงินและเปอร์เซ็นต์

NER เป็นส่วนสำคัญในการดึงข้อมูล (Information Extraction) เพื่อทำให้คอมพิวเตอร์เข้าใจความหมายและบริบทของข้อความ


### **6.2 ความสำคัญของ NER**

* ช่วยสกัด **ข้อมูลเชิงลึก** จากเอกสารจำนวนมาก
* ลดความซับซ้อนของข้อมูลสำหรับงาน **Text Mining**
* สนับสนุนการสร้างระบบ **Question Answering (QA)**
* ใช้ในการเชื่อมโยงข้อมูลกับฐานข้อมูลที่มีโครงสร้าง (Structured Data)
* มีบทบาทสำคัญในงาน **Cybersecurity, Finance, Healthcare, Social Media Analysis**


### **6.3 ขั้นตอนการทำงานของ NER**

#### **Step 1: Text Preprocessing**

* ทำความสะอาดข้อความ (Cleaning)
* Tokenization → แยกประโยคและคำ
* POS Tagging → ระบุชนิดคำ เช่น **Noun, Verb, Adjective**

#### **Step 2: Entity Detection**

* ตรวจหาคำที่อาจเป็นชื่อเฉพาะ
* ใช้กฎภาษาศาสตร์ (Linguistic Rules) เช่น คำขึ้นต้นด้วยตัวพิมพ์ใหญ่

#### **Step 3: Entity Classification**

* จัดประเภทเอนทิตี เช่น PERSON, LOCATION, ORGANIZATION
* ใช้โมเดล Machine Learning หรือ Deep Learning เพื่อทำนาย


### **6.4 วิธีการทำ Named Entity Recognition**

มี 3 แนวทางหลักที่ใช้ในปัจจุบัน:

#### **(1) Rule-Based Approach**

* ใช้ **พจนานุกรม (Gazetteer)** และ **กฎภาษาศาสตร์** ในการจับชื่อเฉพาะ
* เหมาะกับข้อมูลที่เป็น **Domain-Specific** เช่น การแพทย์ กฎหมาย หรือการเงิน

**ข้อดี:**

* ง่ายต่อการทำความเข้าใจ
* ไม่ต้องใช้ข้อมูลฝึกสอนมาก

**ข้อเสีย:**

* ยืดหยุ่นน้อย ถ้าพบคำใหม่ ๆ นอกพจนานุกรมจะล้มเหลว


#### **(2) Machine Learning-Based Approach**

* ใช้เทคนิค **Supervised Learning** โดยฝึกโมเดลกับชุดข้อมูลที่มีการติดป้ายกำกับแล้ว (Labeled Dataset)
* โมเดลที่นิยมใช้:

  * **Hidden Markov Models (HMMs)**
  * **Conditional Random Fields (CRFs)**
  * **Support Vector Machines (SVMs)**
* ต้องทำ **Feature Engineering** เช่น:

  * รูปแบบตัวอักษร → คำขึ้นต้นด้วยตัวพิมพ์ใหญ่
  * POS Tag → ดูว่าคำนั้นเป็น Noun หรือไม่
  * บริบทของคำรอบข้าง

**ข้อดี:**

* ความแม่นยำสูงกว่าวิธี Rule-Based
* ยืดหยุ่นกับคำศัพท์ใหม่ ๆ

**ข้อเสีย:**

* ต้องใช้ข้อมูลฝึกสอนจำนวนมาก
* ต้องออกแบบฟีเจอร์เอง ทำให้ใช้เวลาและแรงงานสูง


#### **(3) Deep Learning-Based Approach**

เทคนิคที่ทันสมัยและแม่นยำที่สุดในปัจจุบัน
ใช้ **Neural Networks** เพื่อเรียนรู้ **Contextual Representations** ของคำโดยไม่ต้องทำ Feature Engineering ด้วยตนเอง
โมเดลยอดนิยม เช่น:

* **BiLSTM-CRF** → ใช้โครงข่ายประสาทแบบ LSTM ร่วมกับ CRF
* **BERT, RoBERTa, GPT, XLM-R** → Transformer-based Models
* **spaCy และ Hugging Face Transformers** → Frameworks ยอดนิยม

**ข้อดี:**

* ความแม่นยำสูงมาก
* ไม่ต้องสร้างฟีเจอร์เอง
* จัดการคำที่ไม่เคยเห็นมาก่อน (Out-of-Vocabulary Words) ได้ดี

**ข้อเสีย:**

* ต้องใช้ทรัพยากรคอมพิวเตอร์สูง
* ต้องใช้ข้อมูลขนาดใหญ่เพื่อฝึกโมเดล


### **6.5 ตัวอย่างการทำงานของ NER**

**ข้อความตัวอย่าง:**

```
"Elon Musk, the CEO of Tesla, visited Bangkok on 20 August 2025."
```

**ผลลัพธ์ที่ได้จาก NER:**

| **Token**      | **Entity Type** |
| -------------- | --------------- |
| Elon Musk      | PERSON          |
| Tesla          | ORG             |
| Bangkok        | GPE             |
| 20 August 2025 | DATE            |

### **6.6 การประเมินคุณภาพของ NER**

การวัดผลลัพธ์ของโมเดล NER นิยมใช้ **Evaluation Metrics** ดังนี้:

* **Precision** → สัดส่วนเอนทิตีที่ทำนายถูกต้อง
* **Recall** → สัดส่วนเอนทิตีที่โมเดลจับได้ครบถ้วน
* **F1-Score** → ค่าเฉลี่ยแบบถ่วงน้ำหนักระหว่าง Precision และ Recall

### **6.7 การประยุกต์ใช้ NER**

1. **การวิเคราะห์ข่าวสาร** → ดึงชื่อบุคคล สถานที่ และองค์กร
2. **Cybersecurity** → ตรวจหาชื่อโดเมน IP หรือ Malware จาก Log
3. **Financial Analytics** → วิเคราะห์รายงานการเงิน และข้อมูลหุ้น
4. **Healthcare** → สกัดชื่อโรค ยา และวิธีรักษาจาก Medical Reports
5. **Chatbots และ Virtual Assistants** → จับชื่อ สถานที่ หรือวันที่ จากคำถามผู้ใช้


### **สรุป**

* **NER** เป็นเทคนิคสำคัญใน NLP ที่ช่วยระบุชื่อบุคคล องค์กร สถานที่ เวลา และข้อมูลสำคัญอื่น ๆ ในข้อความ
* มี 3 แนวทางหลัก: **Rule-Based**, **Machine Learning-Based** และ **Deep Learning-Based**
* ปัจจุบัน **Deep Learning + Transformers** เช่น **BERT** และ **spaCy** ให้ความแม่นยำสูงสุด
* ใช้ประโยชน์ได้ในหลายสาขา เช่น **ข่าวสาร, การเงิน, การแพทย์, Cybersecurity, Chatbots**


## **7. Text Classification**

### **7.1 ความหมายของ Text Classification**

**Text Classification** คือกระบวนการจัดประเภทข้อความ (Documents, Sentences, หรือ Paragraphs) ให้เป็นหมวดหมู่ (Categories / Labels) โดยใช้เทคนิคใน **Natural Language Processing (NLP)** และ **Machine Learning (ML)**

**ตัวอย่าง:**

* จัดหมวดหมู่ **อีเมล** → Spam / Non-spam
* วิเคราะห์ **รีวิวสินค้า** → Positive / Negative / Neutral
* จัดหมวดหมู่ **ข่าวสาร** → การเมือง, กีฬา, บันเทิง



### **7.2 ประเภทของ Text Classification**

Text Classification แบ่งได้เป็นหลายประเภทตามลักษณะของปัญหา:

#### **(1) Binary Classification**

* จัดข้อความออกเป็น **2 กลุ่ม**
* ตัวอย่าง: Spam vs Non-spam, Positive vs Negative

#### **(2) Multi-class Classification**

* จัดข้อความออกเป็น **หลายกลุ่มที่ไม่ซ้อนทับกัน**
* ตัวอย่าง: การจัดหมวดหมู่ข่าว → การเมือง, กีฬา, เทคโนโลยี

#### **(3) Multi-label Classification**

* ข้อความหนึ่งสามารถอยู่ได้ **หลายหมวดหมู่พร้อมกัน**
* ตัวอย่าง: บทความข่าวเดียวกันอาจถูกจัดในหมวด **การเมือง** และ **เศรษฐกิจ**



### **7.3 ขั้นตอนการทำ Text Classification**

#### **Step 1: Data Collection (การเก็บข้อมูล)**

* รวบรวมข้อความจากแหล่งต่าง ๆ เช่น Social Media, News, Emails, Product Reviews
* ข้อมูลควรถูกติดป้ายกำกับ (Labeled Data) เพื่อใช้สร้างโมเดล Supervised Learning

#### **Step 2: Text Preprocessing**

* Tokenization → แบ่งคำ
* Stopword Removal → ลบคำฟุ่มเฟือย
* Stemming / Lemmatization → ทำให้เหลือรากศัพท์
* Lowercasing, Cleaning → ทำให้ข้อมูลเป็นมาตรฐาน

#### **Step 3: Feature Extraction**

* เปลี่ยนข้อความเป็นตัวเลข เช่น:

  * **Bag of Words (BoW)**
  * **TF-IDF**
  * **Word Embeddings** → Word2Vec, GloVe, FastText
  * **Contextual Embeddings** → BERT, GPT

#### **Step 4: Model Training**

เลือกโมเดลในการจำแนกข้อความ เช่น:

* **Traditional Machine Learning Models**

  * Naïve Bayes
  * Support Vector Machines (SVM)
  * Logistic Regression
  * Random Forest
* **Deep Learning Models**

  * Convolutional Neural Networks (CNN)
  * Recurrent Neural Networks (RNN / LSTM / GRU)
  * Transformer-based Models → BERT, RoBERTa, GPT

#### **Step 5: Model Evaluation**

วัดผลลัพธ์ด้วย **Metrics**:

* Accuracy → ความถูกต้องทั้งหมด
* Precision → ความแม่นยำของข้อความที่ถูกจัดหมวด
* Recall → ความครอบคลุมของการจับหมวดหมู่
* F1-score → ค่ากลางระหว่าง Precision และ Recall



### **7.4 เทคนิคยอดนิยมสำหรับ Text Classification**

#### **(1) Naïve Bayes Classifier**

* ใช้หลัก **Bayes’ Theorem**
* เหมาะกับงานที่มีคำจำนวนมาก เช่น Email Spam Detection
* ทำงานได้เร็วและแม่นยำในกรณีข้อมูลไม่ซับซ้อน

**ข้อดี:**

* ใช้ง่าย ทำงานเร็ว
* ต้องการข้อมูลน้อย

**ข้อเสีย:**

* ไม่ดีหาก Feature มีความสัมพันธ์กันสูง



#### **(2) Support Vector Machines (SVM)**

* ใช้การหาขอบเขตที่ดีที่สุด (**Hyperplane**) ในการแยกข้อมูล
* ทำงานได้ดีในข้อมูลที่มีมิติสูง เช่น TF-IDF

**ข้อดี:**

* แม่นยำสูง เหมาะกับ Text Classification
* ทำงานดีในข้อมูลที่มีมิติสูง (High-dimensional Data)

**ข้อเสีย:**

* ใช้เวลาฝึกโมเดลนานถ้าข้อมูลมีขนาดใหญ่



#### **(3) Deep Learning Models**

* ใช้โครงข่ายประสาท (Neural Networks) เพื่อเรียนรู้บริบทและความสัมพันธ์ระหว่างคำ
* เหมาะกับข้อความที่ซับซ้อนและข้อมูลขนาดใหญ่
* ตัวอย่างโมเดลยอดนิยม:

  * **CNN** → ใช้สำหรับ Text Classification โดยจับ Pattern ของ n-Grams
  * **LSTM / GRU** → ใช้กับข้อความลำดับยาว เช่น บทความ
  * **Transformer Models** → เช่น **BERT, RoBERTa, GPT** ให้ความแม่นยำสูงสุด



### **7.5 ตัวอย่างการทำงานของ Text Classification**

**ข้อมูลตัวอย่าง:**

| **Text**                                  | **Label** |
| ----------------------------------------- | --------- |
| “I love this phone, it's amazing!”        | Positive  |
| “This product is terrible and overpriced” | Negative  |
| “The quality is okay, nothing special”    | Neutral   |

**ผลลัพธ์การจำแนกข้อความ:**

* ประโยคที่ 1 → Positive
* ประโยคที่ 2 → Negative
* ประโยคที่ 3 → Neutral



### **7.6 การประเมินผลลัพธ์ของโมเดล**

เพื่อให้โมเดล Text Classification มีประสิทธิภาพสูง ต้องใช้ **Evaluation Metrics** ดังนี้:

| **Metrics**   | **คำอธิบาย**                                   | **สูตร**                                        |
| ------------- | ---------------------------------------------- | ----------------------------------------------- |
| **Accuracy**  | สัดส่วนข้อความที่จัดหมวดถูกต้องทั้งหมด         | (TP + TN) / (TP + TN + FP + FN)                 |
| **Precision** | ความถูกต้องของข้อความที่โมเดลทำนาย             | TP / (TP + FP)                                  |
| **Recall**    | ความสามารถของโมเดลในการดึงข้อความที่เกี่ยวข้อง | TP / (TP + FN)                                  |
| **F1-Score**  | ค่ากลางระหว่าง Precision และ Recall            | 2 × (Precision × Recall) / (Precision + Recall) |



### **7.7 การประยุกต์ใช้ Text Classification**

1. **Email Spam Detection** → จัดหมวดหมู่อีเมลเป็น Spam / Ham
2. **Sentiment Analysis** → วิเคราะห์ความคิดเห็นลูกค้า
3. **News Categorization** → จัดหมวดข่าวอัตโนมัติ
4. **Customer Support Automation** → จัดประเภทคำถามใน Chatbots
5. **Cybersecurity** → จัดประเภท Threat Reports และ Log Files



### **สรุป**

* **Text Classification** เป็นเทคนิคพื้นฐานใน NLP ที่ใช้จัดหมวดหมู่ข้อความ
* มีขั้นตอนสำคัญ: **Data Collection → Preprocessing → Feature Extraction → Model Training → Evaluation**
* โมเดลที่นิยมใช้: **Naïve Bayes, SVM, Deep Learning, Transformers**
* ประยุกต์ใช้ได้ในหลายด้าน เช่น **Email Filtering, Sentiment Analysis, Chatbots, Cybersecurity**

ได้เลยครับ 🙂
จากเอกสาร **“Week 11 NLP and Text Analytics - New\.pdf”** ใน **หัวข้อที่ 8: Word Embeddings & Language Models** ผมจะสรุปเนื้อหาอย่างละเอียด ครอบคลุมแนวคิด วิธีการทำงาน โมเดลที่ใช้ และการประยุกต์ใช้งานใน **Natural Language Processing (NLP)** ดังนี้:


## **8. Word Embeddings & Language Models**

### **8.1 ความหมายของ Word Embeddings**

**Word Embeddings** คือการแปลงคำ (Words) ให้อยู่ในรูปแบบ **เวกเตอร์ตัวเลข (Numeric Vectors)** เพื่อให้โมเดล Machine Learning และ Deep Learning เข้าใจความหมายของคำและความสัมพันธ์ระหว่างคำได้ดียิ่งขึ้น

ต่างจาก **Bag of Words (BoW)** หรือ **TF-IDF** ที่แทนคำด้วยตัวเลขตามความถี่ **Word Embeddings** จะจับ **บริบท (Context)** และ **ความหมายเชิงลึก (Semantic Meaning)** ของคำใน **Space ที่มีมิติสูง (High-dimensional Vector Space)**

**ตัวอย่างแนวคิด:**

```
king - man + woman ≈ queen
```

หมายความว่าการลบ/บวกเวกเตอร์ของคำสามารถสะท้อนความสัมพันธ์เชิงความหมายได้



### **8.2 ความสำคัญของ Word Embeddings**

* ลด **Dimensionality** ของข้อมูลจาก Sparse Matrix ให้เป็น Dense Vector
* ทำให้โมเดลเข้าใจ **Semantic Similarity** ระหว่างคำ
* สนับสนุนงาน **NLP ขั้นสูง** เช่น Machine Translation, Text Generation, Question Answering
* ใช้เป็น **Input Features** สำหรับโมเดล Deep Learning และ Transformer



### **8.3 เทคนิคยอดนิยมสำหรับ Word Embeddings**

#### **(1) Word2Vec**

* พัฒนาโดย **Google**
* ใช้ **Neural Networks แบบตื้น (Shallow Neural Networks)**
* สร้างเวกเตอร์โดยการเรียนรู้จากบริบทของคำ (**Context Window**)
* มี 2 สถาปัตยกรรมหลัก:

  1. **CBOW (Continuous Bag of Words)** → ทำนายคำเป้าหมายจากบริบท
  2. **Skip-gram** → ทำนายบริบทจากคำเป้าหมาย

**จุดเด่น:**

* จับความสัมพันธ์เชิงความหมายของคำได้ดี
* ใช้ทรัพยากรคอมพิวเตอร์น้อย

**ข้อเสีย:**

* ไม่สามารถจัดการกับคำที่ไม่เคยเห็นมาก่อน (**Out-of-Vocabulary Words, OOV**)
* ไม่สามารถเข้าใจบริบทหลายความหมายของคำเดียวกัน (**Polysemy**) ได้ดี



#### **(2) GloVe (Global Vectors for Word Representation)**

* พัฒนาโดย **Stanford University**
* ใช้ **Matrix Factorization** และ **Co-occurrence Statistics** เพื่อสร้างเวกเตอร์
* เรียนรู้จาก **Global Context** ของคำมากกว่า Word2Vec ที่โฟกัสเฉพาะ Local Context

**จุดเด่น:**

* สร้างเวกเตอร์ที่มีความแม่นยำสูง
* เหมาะกับข้อมูลขนาดใหญ่

**ข้อเสีย:**

* ต้องการพลังประมวลผลสูง
* ไม่สามารถรองรับคำที่ไม่เคยเห็นมาก่อนเช่นเดียวกับ Word2Vec



#### **(3) FastText**

* พัฒนาโดย **Facebook AI Research (FAIR)**
* เป็นการต่อยอดจาก **Word2Vec** โดยใช้แนวคิด **Subword Information**
* แทนคำเป็นชุดของ **Character n-Grams** → สามารถสร้างเวกเตอร์ของคำใหม่ที่ไม่เคยเห็นมาก่อน

**จุดเด่น:**

* จัดการคำที่ไม่เคยเห็นมาก่อน (OOV) ได้ดี
* เหมาะกับภาษาเชิงผสมคำ (Morphologically Rich Languages) เช่น ภาษาไทย



#### **(4) Transformer-based Embeddings (BERT, GPT, RoBERTa, etc.)**

โมเดลภายใต้สถาปัตยกรรม **Transformer** เช่น **BERT** และ **GPT** ได้ยกระดับคุณภาพของ Word Embeddings ขึ้นอีกขั้น เพราะสามารถสร้าง **Contextual Embeddings** ที่เปลี่ยนไปตามตำแหน่งและบริบทของคำในประโยค

**ตัวอย่างโมเดลยอดนิยม:**

* **BERT (Bidirectional Encoder Representations from Transformers)**

  * เรียนรู้บริบทแบบสองทิศทาง (**Bidirectional**)
  * เหมาะกับงาน **Text Classification, Named Entity Recognition, Question Answering**
* **GPT (Generative Pretrained Transformer)**

  * เหมาะกับงาน **Text Generation** และ **Conversational AI**
* **RoBERTa, XLNet, ALBERT** → โมเดลพัฒนาต่อยอดจาก BERT

**จุดเด่น:**

* เข้าใจบริบทซับซ้อนได้ดีมาก
* รองรับคำหลายความหมายในบริบทที่แตกต่างกัน

**ข้อเสีย:**

* ใช้ทรัพยากรสูงมาก
* ต้องการข้อมูลและเวลาในการฝึกโมเดลเยอะ


### **8.4 ความแตกต่างระหว่าง Word Embeddings แบบดั้งเดิมและ Contextual Embeddings**

| **คุณสมบัติ**          | **Word2Vec / GloVe / FastText**     | **BERT / GPT (Contextual)**       |
| ---------------------- | ----------------------------------- | --------------------------------- |
| **ประเภทเวกเตอร์**     | คงที่ (Static)                      | เปลี่ยนตามบริบท (Dynamic)         |
| **เข้าใจหลายความหมาย** | ไม่ดี                               | ดีมาก                             |
| **รองรับคำใหม่ (OOV)** | ดี (FastText), แย่ (Word2Vec/GloVe) | ดีถ้าโมเดลมี Subword Tokenization |
| **ความแม่นยำ**         | ปานกลาง                             | สูงมาก                            |
| **ทรัพยากรที่ต้องใช้** | น้อย                                | สูง                               |

### **8.5 การประยุกต์ใช้ Word Embeddings และ Language Models**

1. **Text Classification** → ใช้เวกเตอร์แทนคำเพื่อปรับปรุงประสิทธิภาพโมเดล
2. **Sentiment Analysis** → วิเคราะห์ความรู้สึกเชิงลึกจากบริบท
3. **Named Entity Recognition (NER)** → จับชื่อเฉพาะในข้อความ
4. **Machine Translation** → ใช้ใน Google Translate และ DeepL
5. **Question Answering (QA)** → ใช้ใน Chatbots, Virtual Assistants
6. **Semantic Search** → ปรับปรุงระบบ Search Engines ให้เข้าใจความหมายมากกว่าการจับคีย์เวิร์ด


### **8.6 สรุป**

* **Word Embeddings** คือการแทนคำเป็นเวกเตอร์ตัวเลขที่จับความสัมพันธ์เชิงความหมาย
* เทคนิคยอดนิยม ได้แก่ **Word2Vec, GloVe, FastText** และ **Transformer-based Embeddings**
* **Transformers** เช่น **BERT** และ **GPT** มีความสามารถสูงสุดในการเข้าใจบริบทเชิงลึก
* ใช้ประโยชน์ได้กว้างขวาง เช่น **Text Classification, Sentiment Analysis, Chatbots, Translation**

## **9. Applications of NLP**

### **9.1 บทนำ**

**Natural Language Processing (NLP)** และ **Text Analytics** เป็นเทคโนโลยีที่ช่วยให้คอมพิวเตอร์สามารถ **เข้าใจ, วิเคราะห์, แปลความหมาย และสร้างภาษามนุษย์** ได้ ซึ่งถูกประยุกต์ใช้ในหลายอุตสาหกรรม ไม่ว่าจะเป็น **การตลาด, การแพทย์, การเงิน, Cybersecurity, Social Media, Chatbots** และอื่น ๆ


### **9.2 การประยุกต์ใช้ NLP ในงานจริง**

#### **(1) Sentiment Analysis (การวิเคราะห์ความคิดเห็น)**

* วิเคราะห์ความคิดเห็นเชิงบวก, เชิงลบ และเป็นกลางจากข้อความ
* ใช้ในงาน **Social Media Analytics**, **Customer Feedback** และ **Brand Monitoring**

**ตัวอย่างการใช้งาน:**

* วิเคราะห์ความคิดเห็นจาก Twitter เพื่อประเมินภาพลักษณ์ของแบรนด์
* วิเคราะห์รีวิวสินค้าใน Shopee, Lazada, Amazon

**เทคนิคที่ใช้:**

* **Lexicon-based** → ใช้พจนานุกรมคำเชิงบวก/ลบ
* **Machine Learning-based** → Naïve Bayes, SVM
* **Deep Learning-based** → BERT, RoBERTa, GPT



#### **(2) Spam Detection (การตรวจจับสแปม)**

* ใช้ใน **Email Filtering** และ **SMS Filtering**
* วิเคราะห์เนื้อหาของอีเมลหรือข้อความเพื่อแยก **Spam** และ **Non-Spam**

**เทคนิคที่ใช้:**

* **Bag of Words (BoW)** + **Naïve Bayes**
* **TF-IDF** + **Support Vector Machines (SVM)**
* **Transformer Models** สำหรับระบบกรองสแปมขั้นสูง

**ตัวอย่าง:**

* Gmail ใช้ NLP + Machine Learning ในการกรองสแปมอย่างแม่นยำ
* การกรองข้อความ SMS หลอกลวง เช่น ข้อความธนาคารปลอม



#### **(3) Machine Translation (การแปลภาษาอัตโนมัติ)**

* ระบบแปลภาษาสามารถทำงานได้แบบ **Real-time** และมีความแม่นยำสูง
* ใช้ใน **Google Translate, DeepL, Microsoft Translator**

**เทคนิคที่ใช้:**

* **Statistical Machine Translation (SMT)** → ใช้สถิติความถี่คำ
* **Neural Machine Translation (NMT)** → ใช้ Deep Learning
* **Transformer Models** เช่น **mBERT, MarianMT, GPT** ทำให้คุณภาพการแปลดีขึ้นอย่างมาก

**ตัวอย่าง:**
แปลข้อความจากไทย → อังกฤษ หรืออังกฤษ → จีน ได้แบบบริบทสมบูรณ์



#### **(4) Chatbots & Virtual Assistants (ระบบแชทบอทและผู้ช่วยเสมือน)**

* NLP ทำให้แชทบอทและผู้ช่วยอัจฉริยะเข้าใจและโต้ตอบภาษามนุษย์ได้
* ใช้ใน **บริการลูกค้า, การขาย, การจองบริการ, ระบบสอบถามอัตโนมัติ**

**ตัวอย่างที่ใช้จริง:**

* **ChatGPT** → สนทนาอัจฉริยะ
* **Google Assistant, Siri, Alexa** → ผู้ช่วยส่วนตัว
* **Chatbot ในธนาคาร** → ให้ข้อมูลบัญชีและโปรโมชั่นอัตโนมัติ

**เทคนิคที่ใช้:**

* **Intent Detection** → ใช้ Text Classification เพื่อเข้าใจคำถาม
* **Named Entity Recognition (NER)** → ระบุข้อมูลสำคัญ เช่น ชื่อ, วันที่, สถานที่
* **Dialogue Management** → สร้างการสนทนาที่ต่อเนื่อง


#### **(5) Recommendation Systems (ระบบแนะนำเนื้อหา)**

* ใช้ NLP วิเคราะห์ข้อมูลพฤติกรรมผู้ใช้และข้อความ เพื่อแนะนำสินค้า บทความ หรือวิดีโอ
* ใช้ใน **Netflix, YouTube, TikTok, Shopee, Lazada**

**เทคนิคที่ใช้:**

* **Content-based Filtering** → วิเคราะห์ข้อมูลข้อความของสินค้าหรือบทความ
* **Collaborative Filtering** → ใช้พฤติกรรมผู้ใช้งานคนอื่น ๆ
* **Hybrid Models** → รวม NLP + Deep Learning + Metadata

**ตัวอย่าง:**

* Netflix แนะนำภาพยนตร์ที่เกี่ยวข้องกับสิ่งที่ผู้ใช้เคยดู
* Shopee แนะนำสินค้าที่คล้ายกับสินค้าที่เคยซื้อ


#### **(6) Information Retrieval & Search Engines**

* NLP ทำให้ระบบค้นหามีประสิทธิภาพสูงขึ้น โดยเข้าใจ **ความหมาย (Semantic Search)** ไม่ใช่แค่การจับคีย์เวิร์ด
* ใช้ใน **Google, Bing, Baidu, ElasticSearch**

**เทคนิคที่ใช้:**

* **TF-IDF** + **Cosine Similarity**
* **BM25 Ranking Algorithm**
* **Embedding-based Search** → ใช้ BERT, GPT, Sentence Transformers


#### **(7) Text Summarization (การสรุปเนื้อหาอัตโนมัติ)**

* ใช้ NLP ในการสร้างบทสรุปเนื้อหาอัตโนมัติจากเอกสารยาว ๆ
* ประยุกต์ใช้ใน **ข่าว, งานวิจัย, รายงานทางธุรกิจ**

**วิธีการสรุปเนื้อหา:**

* **Extractive Summarization** → เลือกประโยคสำคัญจากเอกสาร
* **Abstractive Summarization** → เขียนสรุปใหม่โดยใช้ Deep Learning + Transformers

**ตัวอย่าง:**

* Google News ใช้ NLP ในการสร้างบทสรุปข่าว
* ChatGPT สรุปเอกสารและงานวิจัยอัตโนมัติ


#### **(8) Healthcare & Biomedical NLP**

* ใช้ NLP วิเคราะห์ **Electronic Health Records (EHRs)**
* วิเคราะห์ผลตรวจทางการแพทย์, การวินิจฉัยโรค และข้อมูลจากบทความวิจัย

**ตัวอย่างการใช้งาน:**

* สกัดข้อมูลอาการและยาจากบันทึกแพทย์
* วิเคราะห์บทความวิจัยเพื่อค้นหายาใหม่
* ตรวจจับแนวโน้มการระบาดของโรค


#### **(9) Cybersecurity Analytics**

* ใช้ NLP ในการวิเคราะห์ข้อความ Log, Threat Reports, และข่าวสารด้านความปลอดภัย
* ตรวจจับ **Phishing Emails, Malware Signatures, Vulnerability Reports**

**ตัวอย่าง:**

* ระบบตรวจจับ Email หลอกลวงโดยวิเคราะห์ข้อความ
* สกัดข้อมูลจาก Threat Intelligence Feeds


### **9.3 บทสรุปการประยุกต์ใช้ NLP**

| **การประยุกต์**     | **เทคนิคที่ใช้**                | **ตัวอย่างแพลตฟอร์ม**  |
| ------------------- | ------------------------------- | ---------------------- |
| Sentiment Analysis  | TF-IDF, Naïve Bayes, BERT       | Twitter, Facebook      |
| Spam Detection      | BoW, SVM, Transformers          | Gmail, Outlook         |
| Machine Translation | NMT, Transformers               | Google Translate       |
| Chatbots            | Intent Detection, NER, GPT      | Siri, Alexa, ChatGPT   |
| Recommendation      | NLP + Collaborative Filtering   | Netflix, Shopee        |
| Semantic Search     | BERT, Sentence Transformers     | Google Search          |
| Summarization       | Extractive / Abstractive Models | Google News, ChatGPT   |
| Healthcare NLP      | NER, Deep Learning              | EHRs, PubMed           |
| Cybersecurity       | NLP + Threat Intelligence       | Email Security Systems |


### **สรุป**

* NLP และ Text Analytics ถูกประยุกต์ใช้อย่างแพร่หลายในหลายสาขา
* งานสำคัญ เช่น **Sentiment Analysis, Spam Detection, Machine Translation, Chatbots, Summarization**
* เทคโนโลยีหลักที่ทำให้การประยุกต์ใช้งานมีประสิทธิภาพสูงขึ้นคือ **Word Embeddings, Transformers, BERT, GPT**
* แนวโน้มปัจจุบันกำลังมุ่งไปสู่ **Generative AI** และ **Contextualized NLP**


