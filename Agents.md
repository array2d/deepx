## agent 规则
+ This is the only AGENTS.md, there are no recursive AGENTS.md
+ When you are working on a bug, first create a standalone file that reproduces the bug and verify it fails in the expected way. Use this to test if your changes work. Once the change is passing, find an appropriate test file to add the test to and make sure to follow local conventions on the test file.
+ Always respond in 中文,不要回答重复的内容（如我提问中的代码）

## deepx的架构

项目分为3部分
1. 前端。python库的接口风格参考pytorch
2. 编译，调度器，待设计
3. 执行器，使用c++,cuda,metal,omp simd等,实现不同executor的算子

# 关于deepx的细节概念
+ deepx.Tensor仅仅就是一个tensor，不像pytorch的tensor，一个tensor其实包含了自身和梯度2个tensor的数据


贴近pytorch的接口风格，不要增加任何注释，我会手动添加注释

关于doc目录
采用Sphinx构建,使用reStructuredText格式