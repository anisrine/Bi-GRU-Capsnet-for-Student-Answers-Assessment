<html>
<head>

 </head>
<body>
<h3> Bi-GRU-Capsnet for student answers assessment </h3>
  
<p>This is a keras implementation of Bi-GRU-Capsnet for the students answers assessment. 

<p>Ait Khayi, N., & Rus, V. (2019). <a href="http://ml4ed.cc/attachments/Khayi.pdf">Bi-gru capsule networks for student answers assessment</a>.DL4Ed-KDD 2019 


<h3> Requirements </h3>
<p>keras and tensorflow
  
<h3> Data </h3>
<p> DT-Grade : collected from the interaction of students with DeepTutor.
<p> Data Download: <a href="http://deeptutor.memphis.edu/resources.htm">DT-Grade Dataset</a>
<p> Paper:  <a href="https://www.aclweb.org/anthology/W16-0520.pdf">Evaluation Dataset (DT-Grade) and Word Weighting Approach towards
Constructed Short Answers Assessment in Tutorial Dialogue Context</a>. ACL.

<h3>Model Architecture</h3>
<img src="https://i.postimg.cc/qB1tFCrg/capsnet.jpg" alt="Bi-GRU-Capsnet Architecture">
<p> The model consists of the following components : <br>
    <ul>
    <li>Embedding Layer (Glove,Word2vec,Elmo) </li>
    <li>Bi-GRU Layer</li>
    <li>Capsule Layer</li>
    <li> Softmax Layer</li>
</ul>

</body>
</html>


