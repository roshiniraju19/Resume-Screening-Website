import streamlit as st
import pickle
import re
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# Function to clean resume text
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Function to get interview questions based on category
def get_interview_questions(category_id):
    questions_mapping = {
        15: [
            "What is Busy Spinning? Why Should You Use It in Java?",
            "What is Read-Write Lock? Does ConcurrentHashMap in Java Use The ReadWrite Lock? ",
            "How to Make an Object Immutable in Java? Why Should You Make an Object Immutable? ",
            "Describe and compare fail-fast and fail-safe iterators. Give examples.",
            "Why would it be more secure to store sensitive data (such as a password, social security number, etc.) in a character array rather than in a String?",
            "What is the volatile keyword? How and why would you use it?",
            "Compare the sleep() and wait() methods in Java, including when and why you would use one vs. the other.",
            "How can you swap the values of two numeric variables without using any other variables?",
            "When designing an abstract class, why should you avoid calling abstract methods inside its constructor?",
            "How are Java enums more powerful than integer constants? How can this capability be used?",
        ],
        23: [
            "What is the difference between manual testing and automated testing?",
            "How do you identify and report a software bug?",
            "What are the different types of software testing?",
            "What are the different phases of the software testing life cycle?",
            "What are the different levels of testing?",
            "What is a traceability matrix?",
            "What is a test case?",
            "What does a test plan consist of?",
            "What do you mean by Quality Assurance(QA)?",
            "What is Software Configuration Management?",
        ],
        8: [
            "What is the purpose of continuous integration and continuous deployment (CI/CD)?",
            "Explain the concept of infrastructure as code (IaC).",
            "How is DevOps different from agile methodology?",
            "Which are some of the most popular DevOps tools?",
            "How will you approach a project that needs to implement DevOps?",
            "How does continuous monitoring help you maintain the entire architecture of the system?",
            "What is the role of AWS in DevOps?",
            "Why Has DevOps Gained Prominence over the Last Few Years?",
            "What is the Blue/Green Deployment Pattern?",
            "What are the benefits of Automation Testing?",
        ],
        20: [
            "What are the benefits of using Python language as a tool in the present scenario?",
            "Is Python a compiled language or an interpreted language?",
            "What is the difference between a Mutable datatype and an Immutable data type?",
            "How are arguments passed by value or by reference in Python?",
            "What is the difference between a Set and Dictionary?",
            "What is a pass in Python?",
            "Can we Pass a function as an argument in Python?",
            "What is docstring in Python?",
            "What is a dynamically typed language?",
            "What are Built-in data types in Python?",
        ],
        24: [
            "Mention what are the main language or platform used for web-design?",
            "Explain how can you set an image as a background on web pages?",
            "Mention what do you mean by Responsive design on a web page?",
            "Mention what are some bad examples of web design?",
            "Explain what is Information Architecture?",
            "Explain what is a Dreamweaver Template?",
            "Explain what is the difference between “visibility:hidden” and “display:none”?",
            "List out some of the JQuery function used for webpage designing?",
            "As a web-designer while declaring “delete” button what would be your color choice?",
            "Explain how can a developer learn about web design?",
        ],
        12: [
            "Tell me about yourself",
            "What are your strengths and weaknesses?",
            "Tell me about an experience when you faced difficulty at work while working on a project?",
            "Where do you see yourself in the next 5 years?",
            "Would you like to work overtime or odd hours?",
            "Why did you leave your last job?",
            "How do you handle stress, pressure, and anxiety?",
            "Is there anything that makes you different from other candidates?",
            "Tell me about a time when you were not satisfied with your performance?",
            "Why are you interested in this job?",
        ],
        13: [
            "1. Can you explain the key components of the Hadoop ecosystem?",
            "How does HDFS distribute and store data across a cluster of machines?",
            "Explain the MapReduce programming model and its role in Hadoop.",
            "What is the significance of block size in HDFS?",
            "What is Hive, and how does it simplify data processing in Hadoop?",
            "How is Hive different from traditional relational databases?",
            "Can you discuss the purpose of HBase in the Hadoop ecosystem?",
            "Explain the concept of data partitioning in Hadoop.",
            "How does YARN improve the resource management in Hadoop?",
            "Can you provide an example of a scenario where using Hadoop would be beneficial?",
        ],
        3: [
            "What is blockchain, and how does it differ from traditional databases?",
            "Can you explain the concept of consensus algorithms in blockchain?",
            "How are transactions verified and added to the blockchain?",
            "What is the role of smart contracts in blockchain technology?",
            "Can you discuss the differences between public and private blockchains?",
            "How does blockchain ensure security and immutability of data?",
            "Explain the process of mining in a proof-of-work blockchain system.",
            "What challenges and scalability issues are associated with blockchain technology?",
            "How does a decentralized system contribute to the integrity of a blockchain network?",
            "Can you provide examples of real-world applications where blockchain technology is beneficial?",
        ],
        10: [
            "Can you explain the ETL process and its importance in data integration?",
            "What is the difference between ETL and ELT (Extract, Load, Transform)?",
            "How do you handle incremental data extraction to keep the ETL process efficient?",
            "Can you discuss your experience with different ETL tools and technologies?",
            "What strategies do you use for error handling and data quality assurance in ETL processes?",
            "Explain the concept of data profiling and how it is used in ETL development.",
            "How do you optimize the performance of ETL jobs for large datasets?",
            "Can you provide an example of a complex data transformation you implemented in a previous project?",
            "Discuss your approach to designing and maintaining data warehouses in the context of ETL development.",
            "How do you ensure data security and compliance during the ETL process?",
        ],
        18: [
            "Can you provide an overview of your experience in operations management and the industries you've worked in?",
            "How do you approach the development and implementation of operational strategies to improve efficiency and productivity?",
            "Describe a situation where you successfully streamlined operational processes to reduce costs without compromising quality.",
            "What key performance indicators (KPIs) do you consider most important for measuring operational success, and how do you track them?",
            "How do you handle conflict resolution and communication within a diverse operational team?",
            "Can you discuss your experience with supply chain management and inventory control?",
            "Explain how you prioritize and allocate resources to meet organizational goals and deadlines.",
            "Discuss a specific instance where you implemented changes in response to shifting market conditions or industry trends.",
            "How do you ensure compliance with relevant regulations and standards in your operational role?",
            "Share an example of a challenging problem you faced as an operations manager and how you addressed it to achieve a positive outcome.",
        ],
        6: [
            "Can you discuss your experience with data science projects, including specific tools and techniques you've used?" ,
            "Explain the end-to-end process of a typical data science project you've worked on, from problem definition to model deployment." ,
            "How do you approach feature selection and engineering in the context of machine learning projects?" ,
            "Discuss a situation where you had to deal with missing or incomplete data. What strategies did you use to handle it?" ,
            "What machine learning algorithms do you find most effective in various types of data analysis tasks, and why?" ,
            "Can you provide an example of a project where you applied deep learning techniques, and what was the outcome?" ,
            "How do you assess the performance of a machine learning model, and what metrics do you prioritize?" ,
            "Discuss your experience with data visualization tools and techniques. How do you communicate complex findings to non-technical stakeholders?" ,
            "Explain the importance of exploratory data analysis (EDA) in the data science process and share an example of its impact on a project you worked on." ,
            "How do you stay updated on the latest advancements and best practices in the field of data science?",
        ],
        22: [
            "Can you discuss your experience in sales, including specific industries and types of products/services you've sold? ",
            "How do you approach building and maintaining relationships with clients or customers?",
            "Can you provide an example of a challenging sales situation you encountered and how you successfully navigated it?",
            "What strategies do you use to identify and prospect potential clients or customers?",
            "Explain your approach to understanding customer needs and tailoring your sales pitch accordingly.",
            "How do you handle objections and rejections during the sales process?",
            "Can you share a successful sales campaign or deal you closed and the key factors that contributed to its success?",
            "Discuss your experience with using CRM (Customer Relationship Management) tools to manage and track sales activities.",
            "How do you stay informed about industry trends and changes in the market that may impact your sales strategies?",
            "What metrics do you use to measure your sales performance, and how do you continuously work to improve them?",
        ],
        16: [
            "Can you provide an overview of your experience in mechanical engineering and the types of projects you've worked on?",
            "Describe a challenging engineering problem you encountered in a previous role and how you approached solving it.",
            "How do you stay updated on the latest advancements and technologies in the field of mechanical engineering?",
            "Can you discuss your experience with computer-aided design (CAD) software and other engineering tools?",
            "Explain a situation where you had to work collaboratively with cross-functional teams, and what role you played in achieving project goals.",
            "Discuss your familiarity with industry standards and regulations relevant to mechanical engineering.",
            "How do you approach the design and testing of mechanical components to ensure they meet performance and safety requirements?",
            "Can you provide an example of a project where you had to manage competing priorities and tight deadlines?",
            "Discuss your experience with failure analysis and troubleshooting in mechanical systems.",
            "How do you incorporate sustainability and efficiency considerations into your mechanical engineering projects?",
        ],
        1: [
            "Can you provide an overview of your experience and background in the arts, including your education and previous projects?",
            "What inspired you to pursue a career in the arts, and how has your passion evolved over time?",
            "Discuss a specific art project or work you've been involved in, detailing your role and contributions.",
            "How do you approach the creative process from ideation to completion in your artistic work?",
            "Can you discuss your experience with different artistic mediums, such as painting, sculpture, digital art, etc.?",
            "Share your perspective on the intersection of technology and art. How do you incorporate digital tools or new media into your work?",
            "How do you stay inspired and keep up with contemporary art trends and movements?",
            "Discuss your experience collaborating with other artists or working on interdisciplinary projects.",
            "Can you provide examples of how you handle critique and feedback on your artistic work?",
            "What role do you believe the arts play in society, and how does your work contribute to that role?",
        ],
        7: [
            "Can you discuss your experience with database management systems, including specific technologies you've worked with?"
            "How do you approach database design, and what factors do you consider when creating a new database schema?"
            "Explain the differences between relational and non-relational databases. In what situations would you choose one over the other?"
            "Can you provide an example of a complex database query you've written and the problem it addressed?"
            "Discuss your experience with database indexing. How does it impact query performance, and when would you use it?"
            "What is database normalization, and why is it important in database design?"
            "How do you ensure data integrity and consistency in a database?"
            "Can you explain the concept of ACID properties in the context of database transactions?"
            "Discuss your experience with database optimization and performance tuning."
            "How do you approach database security, including user access control and encryption of sensitive data?",
        ],
        11: [
            "Can you provide an overview of your experience in electrical engineering and the specific areas within the field you've worked on?",
            "Discuss a challenging project you've been involved in, detailing your role and contributions.",
            "How do you stay updated on the latest developments and advancements in electrical engineering?",
            "Explain your experience with designing and implementing electrical systems, including any specific tools or software you use.",
            "Can you discuss your familiarity with different electrical components and their applications?",
            "Describe a situation where you had to troubleshoot and solve a complex electrical problem.",
            "How do you approach the design and testing of electrical circuits to ensure they meet performance and safety requirements?",
            "Discuss your experience working with power systems and your understanding of electrical regulations and standards.",
            "Can you provide an example of a project where you collaborated with other engineering disciplines, such as mechanical or civil engineering?",
            "How do you balance competing priorities and tight deadlines in your electrical engineering projects?",
        ],
        14: [
            "Can you discuss your background and experience in the health and fitness industry, including any relevant certifications?",
            "What inspired you to pursue a career in health and fitness, and how has your passion evolved over time?",
            "Describe your approach to creating personalized fitness plans for individuals with varying goals and fitness levels.",
            "How do you stay informed about the latest trends, research, and best practices in health and fitness?",
            "Can you share a success story of a client you've worked with, highlighting the positive impact of your guidance on their health and fitness journey?",
            "Discuss your experience with different exercise modalities and your ability to adapt workouts for diverse populations.",
            "How do you address common challenges clients face in adhering to fitness plans, and what strategies do you employ to keep them motivated?",
            "Explain your understanding of nutrition's role in overall health and fitness. How do you integrate dietary guidance into your programs?",
            "Can you discuss any experience you have with group fitness instruction or leading fitness classes?",
            "How do you approach the assessment of clients' physical fitness and health conditions before developing a fitness plan?",
        ],
        19: [
            "Can you discuss your experience in project management and your specific role within PMO structures? ",
            "Explain the key responsibilities of a PMO and how it contributes to the success of projects and the overall organization. ",
            "How do you ensure alignment between project goals and organizational strategy within the PMO?",
            "Discuss your experience with project portfolio management and how you prioritize projects within a portfolio.",
            "Can you provide an example of a challenging project or program you managed within a PMO, detailing the strategies you employed to overcome obstacles? ",
            "Explain your approach to risk management within a PMO context. How do you identify, assess, and mitigate risks throughout the project lifecycle?",
            "Discuss your experience with project reporting and how you communicate project status and key metrics to stakeholders.",
            "How do you foster collaboration and communication among different project teams within the PMO?",
            "Explain your understanding of governance frameworks and how they are applied in the context of a PMO.",
            "Can you share examples of how you have contributed to continuous improvement initiatives within the PMO?",
        ],
        4: [
            "Can you discuss your experience as a business analyst, including specific industries and types of projects you've worked on?",
            "How do you gather and document business requirements from stakeholders, and what techniques do you use for requirements elicitation?",
            "Explain your process for analyzing and documenting business processes. How do you identify areas for improvement?",
            "Can you provide an example of a project where you played a key role in translating business needs into technical requirements?",
            "Discuss your experience with data analysis and how you use it to inform business decisions.",
            "How do you ensure the alignment of business objectives with project deliverables and outcomes?",
            "Explain your approach to stakeholder management and how you handle conflicting requirements from different parties.",
            "Can you discuss your experience with modeling tools and techniques, such as process diagrams, data flow diagrams, or use cases?",
            "How do you stay informed about industry trends and changes that may impact business processes and requirements?",
            "Can you share an example of a challenging problem you encountered as a business analyst and how you addressed it to achieve a positive outcome?",
        ],
        9: [
            "Can you discuss your experience as a .NET developer, including specific versions and frameworks you've worked with?",
            "How do you approach application architecture and design when working with the .NET framework?",
            "Explain the differences between ASP.NET Web Forms and ASP.NET MVC. When would you choose one over the other?",
            "Can you provide examples of common design patterns you've used in .NET development, such as MVC or Singleton?",
            "Discuss your experience with database access in .NET applications. How do you work with ADO.NET or Entity Framework?",
            "Explain the concept of .NET Core and its significance in modern application development.",
            "Can you discuss your familiarity with asynchronous programming in .NET, including the use of async/await?",
            "Discuss your experience with version control systems, especially Git, in the context of .NET development.",
            "How do you handle security considerations, such as authentication and authorization, in .NET applications?",
            "Can you share an example of a challenging bug you encountered in a .NET project and how you debugged and resolved it?",
        ],
        2: [
            "Can you discuss your experience with automation testing, including specific testing frameworks and tools you've used?",
            "Explain the benefits and challenges of automated testing compared to manual testing.",
            "How do you determine which test cases are suitable for automation, and what criteria do you consider?",
            "Discuss your experience with testing frameworks such as Selenium, JUnit, TestNG, or others commonly used in automation testing.",
            "Can you provide an example of a complex test scenario you automated and the challenges you encountered during the process?",
            "Explain the concept of data-driven testing and how you implement it in your automated testing strategy.",
            "Discuss your familiarity with continuous integration and how automation testing fits into the CI/CD pipeline.",
            "How do you handle dynamic or frequently changing user interfaces in automated testing?",
            "Can you share your approach to handling and reporting test failures in automated test suites?",
            "Discuss your experience with performance testing automation, including tools like JMeter or Gatling if applicable.",
        ],
        17: [
            "Can you discuss your experience as a network security engineer, including specific technologies and security protocols you've worked with?",
            "Explain the key principles of network security and how they contribute to overall cybersecurity.",
            "Discuss your experience with firewall configurations and rules. How do you determine the appropriate level of access control for different network segments?",
            "Can you provide examples of security measures you've implemented to protect against common network attacks such as DDoS, phishing, or malware?",
            "Explain your understanding of VPN (Virtual Private Network) technologies and how they enhance network security.",
            "Discuss your familiarity with intrusion detection and prevention systems (IDS/IPS). How do you use them to monitor and respond to security threats?",
            "How do you approach network vulnerability assessments and penetration testing to identify and address potential security risks?",
            "Can you discuss your experience with security information and event management (SIEM) systems for log analysis and incident response?",
            "Explain the concept of network segmentation and how it enhances security in complex networks.",
            "How do you stay updated on the latest trends and developments in network security, and can you provide examples of how you've applied new knowledge to enhance security measures?",
        ],
        21: [
            "Can you discuss your experience as a SAP developer, including specific modules and versions you've worked with? ",
            "Explain the typical lifecycle of an SAP development project, from requirement gathering to deployment.",
            "Discuss your experience with ABAP programming language, and how you use it in SAP development.",
            "Can you provide examples of SAP modules you've customized or enhanced to meet specific business requirements?",
            "Explain your approach to debugging and troubleshooting SAP applications.",
            "Discuss your familiarity with SAP Fiori and how it contributes to the user experience in SAP applications.",
            "How do you integrate SAP applications with other systems or third-party tools?",
            "Can you discuss your experience with SAP HANA and in-memory computing within the SAP landscape?",
            "Explain your understanding of SAP security measures and how you ensure secure coding practices in SAP development.",
            "How do you approach performance optimization in SAP development projects? Can you provide examples of improvements you've made?",
        ],
        5: [
            "Can you provide an overview of your experience as a civil engineer, including the types of projects you've worked on?",
            "Explain your process for conducting site assessments and feasibility studies for construction projects.",
            "Discuss your experience with civil engineering design software and tools, such as AutoCAD or Civil 3D.",
            "Can you provide examples of projects where you managed the design and construction of infrastructure, such as roads, bridges, or buildings?",
            "Explain your approach to project management in civil engineering, including how you handle budgets, timelines, and stakeholder communication.",
            "Discuss your familiarity with relevant codes and standards in civil engineering and how you ensure compliance in your projects.",
            "How do you approach environmental considerations in civil engineering projects, such as sustainability and impact assessments?",
            "Can you discuss your experience with geotechnical engineering and how it contributes to the foundation design in construction projects?",
            "Explain your understanding of risk management in civil engineering, especially in mitigating potential issues during construction.",
            "How do you stay informed about new technologies and advancements in civil engineering, and how do you apply them in your work?",
        ],
        0: [
            "Can you provide an overview of your experience as an advocate, including the types of cases you've handled?",
            "Explain your approach to legal research and staying updated on changes in laws and regulations.",
            "Discuss your experience with various legal areas such as civil, criminal, family, or corporate law.",
            "Can you provide examples of cases where you successfully represented clients and achieved positive outcomes?",
            "Explain your process for preparing legal documents, including briefs, contracts, or other relevant legal paperwork.",
            "Discuss your approach to client communication and how you ensure clients are well-informed about their legal matters.",
            "How do you handle challenging situations or disputes in the courtroom, and what strategies do you use to build a strong case?",
            "Can you discuss your experience with alternative dispute resolution methods, such as mediation or arbitration?",
            "Explain your understanding of ethical considerations and professional conduct in the legal profession.",
            "How do you manage your workload and prioritize cases to ensure effective representation for your clients?",
        ],

    }
    return questions_mapping.get(category_id, [])

# Function to display the main content
def main():
    # Set page title and description
    st.title("Career Crafter - A Resume Screening Website")
    st.subheader("Upload your resume and get job predictions along with interview questions!")

    # Upload file and submit button
    uploaded_file = st.file_uploader('Upload Resume (PDF or TXT)', type=['txt', 'pdf'])
    submit_button = st.button("Submit")

    # If submit button clicked
    if submit_button:
        if uploaded_file is None:
            st.error("No file is attached.")
            return  # Stop execution if no file is attached

        try:
            # Read and decode resume text
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        # Clean the resume text
        cleaned_resume = clean_resume(resume_text)

        # Check if words in the resume are less than 20
        if len(cleaned_resume.split()) < 20:
            st.error("CV is invalid. Please upload a valid resume with sufficient content.")
        else:
            # Transform the cleaned resume text and make job predictions
            input_features = tfidfd.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",

            }

            # Get category name and display prediction
            category_name = category_mapping.get(prediction_id, "Unknown")
            st.success(f"Predicted Category: {category_name}")

            # Get interview questions based on predicted category
            interview_questions = get_interview_questions(prediction_id)

            # If questions are available, display them
            if interview_questions:
                st.subheader("\nInterview Questions:")
                for idx, question in enumerate(interview_questions, 1):
                    st.write(f"{idx}. {question}")

# Run the main function
if __name__ == "__main__":
    main()