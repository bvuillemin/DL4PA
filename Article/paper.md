---
title: 'DL4PA: A deep learning framework for process analysis'
tags:
  - Process mining
  - deep learning
  - framework
  - data science
  - modular
  - python
authors:
  - name: Benoit Vuillemin
    orcid: 0000-0002-6929-492X
    corresponding: true
    affiliation: 1
  - name: Frédéric Bertrand
    orcid: 0000-0002-0837-8281
    affiliation: 1

affiliations:
 - name: University of Technology of Troyes, Troyes, France
   index: 1
date: 09 January 2023
bibliography: paper.bib

---

# Summary 

In the field of process analysis, more and more deep learning projects are emerging. However, databases in this domain are often dense and complex, inducing the implementation of complex pre-processing. To address this, this article proposes an open-source framework in Python that is modular, scalable and upgradeable. Its pre-processing architecture and available tools allow to build and upgrade a deep learning project more quickly while keeping the creative freedom inherent to research.


# Statement of need

In process analysis, data are often heterogeneous and very large, which leads to several problems regarding their pre-processing. As a result, most deep learning projects in this domain do this pre-processing by hand through a custom implementation. Thus, projects are often implemented from scratch and the code may not be interpretable. This limits the prospects for evolution and makes these projects difficult to adapt to new datasets and interpret by other research teams.

This framework aims to address this issue by proposing a modular and scalable framework. Here, pre-processing is separated into distinct actions, operated by combinable and interchangeable instances of classes. Moreover, new instances of classes can be created to adapt the framework to the desired objective.

This paper details the precise problems inherent to the use of deep learning in process analysis, then the global operation of the framework. It allows to build projects more easily, with ready-to-use methods, and to make the code much more readable and interpretable.

# Proposal

`DL4PA`, short for 'Deep Learning for Process Analysis', is made in Python 3 and using the NumPy [@harris_array_2020] and pandas [@mckinney_data_2010] libraries. TensorFlow [@martin_abadi_tensorflow_2015] is required to run the provided neural network but is not required otherwise.

The three principles of this framework are **modularity, scalability, and upgradeability**.
Its key actions are the following:
- Edit cases, through `editor` objects,
- Encode the data, through `encoder` objects,
- Separate the data into inputs and outputs, through `data preparator` objects,
- Finally, train the deep learning model.

![Workflow of the framework\label{fig:workflow}](Workflow.pdf)

A representation of the framework's operation can be seen in \autoref{fig:workflow}. The framework has two execution modes:
- Offline, storing the results in files, thus avoiding processing time,
- Online, avoiding storing additional files, useful if the databases are already very heavy in memory. To avoid filling too much RAM, parameters are built in the framework to limit the number of objects to store in memory.

The different components of the framework are modular. That includes the editors, encoders, data preparators and even the neural network models. Each component can be swapped with another without disrupting the rest of the project.

Also, editors can be combined, as do encoders. This means that it is easy to add objects from these classes and make them work, making this framework highly upgradable. Several basic editors and encoders are already present in the proposed framework, but the user is free to create new ones and combine them.

Finally, the input databases are loaded in chunks, the size of which can be defined by the user, thus avoiding memory issues when processing large volumes of data. This, in addition to the online mode, allows the scalability of the framework.

This framework is designed to start a project without having to code everything from scratch and to implement and try new pre-processing and learning techniques more easily than with a hand-made project. It is not intended to provide all the tools needed to do deep learning, but rather the basic architecture to structure a project. This framework is also meant to be updated over time, and can welcome contributions from other researchers to enrich and optimize it.

# Acknowledgements

We acknowledge the financial support of LiveJourney, now acquired by QAD Inc.

# References