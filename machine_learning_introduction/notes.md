- What machine learning is and why it's so important in today's world
- The historical context of machine learning
- The data science process
- The types of data that machine learning deals with
- The two main perspectives in ML: the statistical perspective and the computer science perspective
- The essential tools needed for designing and training machine learning models
- basics of Azure ML
- Distinction between models and algorithms
- The basics of linear regression model
- distinction between parametric vs non-parametric functions
- the distinction between classical machine learning vs deep learning
- the main approaches to machine learning
- the trade-offs that come up when making decisions about how to design and train machine learning models.

## What is Machine Learning

> Machine learning is a data science technique used to extract patterns from data, allowing computers to identify related data, and forecast future outcomes, behaviours and trends.

## Applications of Machine Learning

The applications of machine learning are extremely broad! And the opportunities cut across industry verticals. Whether the industry is healthcare, finance, manufacturing, retail, government or education.
A few examples include:-

- Automate the recognition of disease
- Recommend next best actions for individual care plans
- Enable personalized, real-time banking experience with chatbots
- Identify the next best action for the customer
- capture, prioritize, and route service requests to the correct employee and improve response times.

## The Data Science Process

Big data has become part of the lexicon of organizations worldwide, as more and more organizations look to leverage data to drive informed business decisions. With this evolution in business decision-making, the amount of raw data collected, along with the number and diversity of data sources is growing at an astounding rate. This data presents enormous potential.

Raw data, however is often noisy and unreliable and may contain missing values and outliers. Using such data for modelling can produce misleading results. For the data scientist, the ability to combine large, disparate data sets into a format more appropriate for analysis is an increasingly crucial skill.

The data science process typically starts with collecting and preparing the data before moving on to training, evaluating and deploying a model.

Collect Data -> Prepare Data -> Train Model -> Evaluate Model -> Deploy Model

## The Types of Data

- Numerical
- Time-Series
- Categorical
- Text
- Image

### Tabular Data

In machine learning, the most common type of data you'll encounter is tabular data - that is, data that is arranged in a data table.

It is important to know that in machine learning, we ultimately always work with numbers or specifically _vectors_.

> A vector is simply an array of numbers, such as `(1,2,3)` - or a nested array that contains other arrays of numbers such as `(1,2,(1,2,3))`

Vectors are used heavily in machine learning.

## Scaling Data

Scaling data means transforming it so that the values fit within some range or scale, such as 0-100 or 0-1. There are a number of reasons why it is a good idea to scale your data before feeding it into a machine learning algorithm.

Two common approaches to scaling data includes **standardization** and **normalization**

### Standardization

> **Standardization** rescales data so that it has a mean of 0 and a standard deviation of 1.

The formula for this is

> (𝑥 − 𝜇)/𝜎

### Normalization

> Normalization rescales the data into the range [0,1]

The formula for this is

> (𝑥 −𝑥𝑚𝑖𝑛)/(𝑥𝑚𝑎𝑥 −𝑥𝑚𝑖𝑛)

## Encoding Categorical Data

There are two common approaches for encoding categorical data: **ordinal encoding** and **one hot encoding**.

### Ordinal Encoding.

In ordinal encoding, we simply convert the categorical data into integer codes ranging from `0` to `(number of categories -1)`. One of the potential drawbacks to this approach is that it implicitly assumes order across the categories.

### One-Got Encoding

**One-hot encoding** is a very different approach. In one-got encoding, we transform each categorical value into a column. If there are n categorical values, n new columns are added. If an item belongs to a category, the column representing that category gets the value `1` and all other columns get the value `0`. One drawback of one-hot encoding is that it can potentially generate a very large number of columns.

## Image Data

The color or each pixel is represented with a set of values:

- In **grayscale images**, each pixel can be represented by a **single** number, which typically ranges from 0 to 255.
- In **colored images**, each pixel can be represented by a vector of three numbers (each ranging from 0 to 255) for the three primary color channels: red, green, and blue.

### Encoding an Image

We need to know the following three things about an image to reproduce it:

- Horizontal position of each pixel
- Vertical position of each pixel
- Color of each pixel

Thus, we can fully encode an image numerically by using a vector with three dimensions. The size of the vector required for any given image would be the `height * width * depth` of that image.

### Other Preprocessing Steps

In addition to encoding an image numerically, we may also need to do some other preprocessing steps. Generally, we would want to ensure that the input images have a _uniform aspect ratio_ (e.g. by making sure all of the input images are square in shape) and are normalizes (e.g subtract mean pixel value in a channel from each pixel value in that channel). Some other preprocessing operations we might want to do to clean the input images include rotation, cropping, resizing, denoising and centering the image.

## Text Data

Text is another example of a data type that is initially non-numerical and that must be processed before it can be fed into a machine learning algorithm. Common tasks that we might do as part of this processing include:

### Normalization

One of the challenges that can come up in text analysis is that there are often multiple forms that mean the same thing. Text normalization is the process of transforming a piece of text into a canonical(official) form.

_Lemmatization_ is an example of normalization. A lemma is the dictionary form of a word and lemmatization is the process of reducing multiple inflections to that single dictionary form.

In many cases, you may also want to remove _stop words_. Stop words are high-frequency words that are unnecessary(or unwanted) during the analysis.

Tokenized the text (i.e., split each string of text into a list of smaller parts or tokens), removed stop words (`the`) and standardized spelling

### Vectorization

After we have normalized the text, we can take the next step of actually encoding it in a numerical form. The goal here is to identify the particular features of the text that will be relevant to us for the particular task we want to perform- and then get those features extracted in a numerical form that is accessible to the machine learning algorithm. ypically this is done by text vectorization—that is, by turning a piece of text into a vector. Remember, a vector is simply an array of numbers—so there are many different ways that we can vectorize a word or a sentence, depending on how we want to use it. Common approaches include:

- Term Frequency-Inverse Document Frequency (TF-IDF) vectorization
- Word embedding, as done with Word2vec or Global Vectors (GloVe)

## The Tools for Machine Learning

A typical machine learning ecosystem is made up of three main components:

- **Libraries** - A library is a collection of pre-written(and compiled) code that you can make use of in your own project.
- **Development environments** - A _development environment_ is a software application(or sometimes a group of applications) that provide a whole suite of tools designed to help you(as the developer or machine learning engineer) build out your projects.
- **Cloud services**

### Core Framework and Tools

- **Python** - is a very popular high-level programming language that is great for data science. Its ease of use and wide support within popular machine learning platforms, coupled with a large catalog of ML libraries, has made it a leader in this space.
- **Pandas** - is an open-source Python library designed for analyzing and manipulating data. It is particularly good for working with tabular data and time-series data.
- **Numpy** - like Pandas, is a Python library. NumPy provides support for large, multi-dimensional arrays of data, and has many high-level mathematical functions that can be used to perform operations on these arrays.

### Machine Learning and Deep Learning

- **Scikit-Learn** - is a Python library designed specifically for machine learning. It is designed to be integrated with other scientific and data-analysis libraries, such as NumPy, SciPy, and matplotlib
- **Apache Spark** - is an open-source analytics engine that is designed for cluster-computing and that is often used for large-scale data processing and big data.
- **TensorFlow** - is a free, open-source software library for machine learning built by Google Brain.
- **Keras** - is a Python deep-learning library. It provide an Application Programming Interface (API) that can be used to interface with other libraries, such as TensorFlow, in order to program neural networks. Keras is designed for rapid development and experimentation.
- **Pytorch** - is an open source library for machine learning, developed in large part by Facebook's AI Research lab. It is known for being comparatively easy to use, especially for developers already familiar with Python and a Pythonic code style.

### Data Visualization

- **Plotly** - is not itself a library, but rather a company that provides a number of different front-end tools for machine learning and data science—including an open source graphing library for Python.
- **Matplotlib** - is a Python library designed for plotting 2D visualizations. It can be used to produce graphs and other figures that are high quality and usable in professional publications. You'll see that the Matplotlib library is used by a number of other libraries and tools, such as SciKit Learn (above) and Seaborn (below). You can easily import Matplotlib for use in a Python script or to create visualizations within a Jupyter Notebook.
- **Seaborn** - is a Python library designed specifically for data visualization. It is based on matplotlib, but provides a more high-level interface and has additional features for making visualizations more attractive and informative.
- **Bokeh** - is an interactive data visualization library. In contrast to a library like matplotlib that generates a static image as its output, Bokeh generates visualizations in HTML and JavaScript. This allows for web-based visualizations that can have interactive features.

## Cloud Services for Machine Learning

A typical cloud service for machine learning provides support for managing the core assets involved in machine learning projects.

- **Datasets** - Define, version, and monitor datasets used in machine learning runs.
- **Experiments/Runs** - Organize machine learning workloads and keep track of each task executed through the service.
- **Pipelines** - Structured flows of tasks to model complex machine learning flows.
- **Models** - Model registry with support for versioning and deployment to production.
- **Endpoints** - Expose real-time endpoints for scoring as well as pipelines for advanced automation.

Machine learning cloud services also need to provide support for managing the resources required for running machine learning tasks.

- **Compute** - Manage compute resources used by machine learning tasks.
- **Environments** - Templates for standardized environments used to create compute resources.
- **Datastores** - Data sources connected to the service environment (e.g. blob stores, file shares, Data Lake stores, databases).

## Models vs Algorithms

- **Models** are the specific representations learned from data
- **Algorithms** are the processes of learning

Machine learning models are outputs or specific representations of algorithms that run on data. A model represents what is learned by a machine learning algorithm on the data.