
WebRTC（Web实时通信）是一种基于网页浏览器的 **实时通信技术**，它允许浏览器之间直接进行音频、视频和数据的传输，而无需任何插件或第三方应用程序的支持。

WebRTC已经成为现代Web应用程序中常用的通信标准之一。

chatgpt: 鉴于WebRTC的成熟度和广泛应用，它仍然具有良好的前景，并可能继续在2023年以及之后的时间里经常使用。


>为什么不用 websocket? </br></br>
当带宽不足的时候 websocket 就没法工作了，而 WebRTC 可以把音视频码流降下来，防止网络拥塞把网络打死，关键点是他可以对网络带宽做更准确、更及时的评估预测，再根据预测结果增大或减少码流，使之总是有最好的服务质量。</br></br>websocket底层用的TCP协议，他也有带宽评估，但他评估带宽拥塞会产生巨大网络抖动，音视频质量会受到很大影响。


评论区很有意思（大佬关于通信的理解）：https://www.zhihu.com/question/391589748/answer/1190518209


## 示例

从用户摄像头中获取视频流：

[WebRTC samples getUserMedia](https://webrtc.github.io/samples/src/content/getusermedia/gum/)




----------

参考资料：
- [WebRTC - 教程](https://www.jc2182.com/webrtc/webrtc-jiaocheng.html)
- [教程2](https://webrtcforthecurious.com/zh/)
- [WebRTC predictions for 2023](https://bloggeek.me/webrtc-predictions-2023/#h-preparing-for-a-rocky-year) [中文版](https://zhuanlan.zhihu.com/p/580146138)
- chatgpt