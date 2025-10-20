// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-navin-kumar-m",
    title: "Navin Kumar M",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-posts",
          title: "Posts",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/post/";
          },
        },{id: "nav-projects",
          title: "Projects",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/project/";
          },
        },{id: "nav-repos",
          title: "Repos",
          description: "My open source repositories",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repo/";
          },
        },{id: "nav-resume",
          title: "Resume",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/resume/";
          },
        },{id: "post-feature-fetch-optimization",
        
          title: "Feature Fetch Optimization",
        
        description: "Discover how I engineered a dramatic 8× speedup in feature fetching at Branch International through smart system design and hands-on optimizations. Dive into the actionable strategies and real-world impact behind this major performance boost!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/post/2025/feature-fetch-optimization/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-started-my-undergraduate-studies-at-vellore-institute-of-technology-i-pursued-a-b-tech-in-computer-science-amp-amp-engineering-with-a-specialization-in-artificial-intelligence-amp-amp-machine-learning",
          title: 'Started my undergraduate studies at Vellore Institute of Technology. I pursued a B.Tech...',
          description: "",
          section: "News",},{id: "news-began-a-research-internship-at-vellore-institute-of-technology-focusing-on-distributed-training-and-inference-i-built-a-distributed-training-and-inference-workspace-on-ibm-powerpc-ppc64le-using-ray-which-allowed-15-concurrent-model-experiments",
          title: 'Began a Research Internship at Vellore Institute of Technology, focusing on distributed training...',
          description: "",
          section: "News",},{id: "news-started-my-internship-as-a-software-engineer-machine-learning-at-branch-international-developed-an-sms-cashflow-and-balance-classification-model-0-99-auc-3x-coverage-engineered-credit-bureau-features-6-auc-gain-and-achieved-99-5-data-consistency-with-thorough-qa-qc",
          title: 'Started my internship as a Software Engineer - Machine Learning at Branch International....',
          description: "",
          section: "News",},{id: "news-graduated-from-vellore-institute-of-technology-with-a-b-tech-in-computer-science-amp-amp-engineering-spec-in-ai-amp-amp-ml-with-8-92-cgpa-honored-to-receive-the-best-final-year-project-award-for-my-work-on-an-on-premise-ai-learning-platform-relevant-coursework-machine-learning-deep-learning-data-structures-algorithms-and-software-engineering",
          title: 'Graduated from Vellore Institute of Technology with a B.Tech in Computer Science &amp;amp;amp;...',
          description: "",
          section: "News",},{id: "news-excited-to-join-branch-international-as-a-software-engineer-machine-learning-b1-driving-an-neural-modeling-effort-to-extract-credit-signals-from-sms-based-financial-behavior-built-fully-automated-training-infrastructure-and-accelerated-feature-fetch-process-also-developed-a-credit-model-for-a-premium-user-segment-users",
          title: 'Excited to join Branch International as a Software Engineer – Machine Learning B1!...',
          description: "",
          section: "News",},{id: "projects-ai-malware-system",
          title: 'AI Malware System',
          description: "Fast API Endpoint!. This project examines, analyses the malware statically &amp; dynamically using conventional strategies and also apply machine learning algorithms lke lightgbm, svm and deep learning algorithms like CoAtNet, LSTM. FrontEnd App is Antivirus built on Tauri",
          section: "Projects",handler: () => {
              window.location.href = "/projects/ai-antivirus/";
            },},{id: "projects-chart-derendering-benetech",
          title: 'Chart Derendering - Benetech',
          description: "Derendering Charts and Plots to make them accessible for visually impaired people. This project was done as a part of the Benetech - Making Graphs Accessible Kaggle Competition. MatCha (Pix2Struct) model is used for derendering line, vertical bar, and horizontal bar charts. icevision Faster RCNN model is used for derendering dot and scatter plots.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/benetech-chart-derendering/";
            },},{id: "projects-deepsee-surveillance-ai",
          title: 'DeepSee Surveillance AI',
          description: "Real Time Detection of Anomalous Activity From Videos (mainly crime actvity). Images of the video is trained using AutoEncoder to get the imtermediate feature representation of image &amp; appliend svm model for the bag of such features to detect the anomaly &amp; LSTM to detect the type of Anomaly.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/deepsee-crime/";
            },},{id: "projects-face-surveillance-system",
          title: 'Face Surveillance System',
          description: "Face Recognition from Crowd by using yolov7 .Extracting the faces from a video/image/live source, which is then passed to the custom facenet network in order to recognize the peoples",
          section: "Projects",handler: () => {
              window.location.href = "/projects/face-recognition/";
            },},{id: "projects-malware-detection-web-app",
          title: 'Malware-Detection Web App',
          description: "Malware Analysis using Deep Learning &amp; Machine Learning deployed on AWS cloud. ML &amp; DL algorithms was written in Python, Server Part written in node.js",
          section: "Projects",handler: () => {
              window.location.href = "/projects/malware-detection-web/";
            },},{id: "projects-parallel-amp-distributed-ml-workspace",
          title: 'Parallel &amp;amp; Distributed ML Workspace',
          description: "Documentation of Setting up Parallel &amp; Distributed ML Workspace in your systems. And to work seamlessly without error. Package to easily setup a environment in the group of systems",
          section: "Projects",handler: () => {
              window.location.href = "/projects/parallel-distributed/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%70%65%72%73%6F%6E%61%6C.%6D%6E%6B@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/cosmic-heart", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/mnk-navin", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
