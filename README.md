# Efficient Caption Refinement Networks for Dense Video Captioning

Video captioning aims to generate text sentences that describe a given video in an optimal form. With the growth of video platforms such as YouTube, video captioning has extensively been studied as a core technology for video processing due to its wide applicability. State-of-the-art approaches for video captioning have mostly regarded the task as a one-way network that generates from a video to a sentence. However, considering the inverse process that generates the video features from the generated captions, since it is obvious that the reliable captions should be able to reconstruct the original features for the video. Moreover, recent approaches have low memory efficiency by using a large number of layers or additional memory networks to create better captions. To this end, we propose a novel deep learning framework for dense video captioning. Our model firstly generates an initial caption from the pre-extracted video features with the conventional encoder-decoder architecture, and reconstructs the original video feature from the generated caption with another encoder-decoder network. The caption is refined to be regenerated through the same network that created the initial caption. The experimental results show that our model is comparable in captioning performance to methods that use a larger number of parameters, and requires less training time by simply matching the cycle consistency between the original video feature and the reconstructed video feature without using additional memory

## Structure...
<img src="https://user-images.githubusercontent.com/52390523/86232119-79d6d180-bbce-11ea-9899-f77427f2855d.png" width="90%"></img>

## Model Output.........
<img src="https://user-images.githubusercontent.com/52390523/86232209-94a94600-bbce-11ea-9a93-9ecffb67321b.png" width="90%"></img>
<img src="https://user-images.githubusercontent.com/52390523/86232228-996dfa00-bbce-11ea-9127-9e71157b8f45.png" width="90%"></img>
<img src="https://user-images.githubusercontent.com/52390523/86232310-b73b5f00-bbce-11ea-8173-daf7903396e9.png" width="90%"></img>


