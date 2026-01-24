import streamlit as st
import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 导入核心组件
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.tools import render_text_description
from langchain_core.runnables import RunnableLambda

# ================== 页面配置 =======================
st.set_page_config(page_title='Qwen 极简版', layout='wide')
st.title("🤖 Qwen 智能助手 (手动驱动)")

# ================== 左边栏配置 =======================
with st.sidebar:
    st.header("配置参数")
    api_key = st.text_input('Alibaba DashScope API Key', type='password')
    model_name = st.selectbox('选择模型', ('qwen-max', 'qwen-plus', 'qwen-turbo'))
    temperature = st.slider('Temperature', 0.0, 1.0, value=0.1, step=0.1)
    if st.button('清空历史'):
        st.session_state.clear()
        st.rerun()

# ================== 逻辑初始化 =======================
message_history = StreamlitChatMessageHistory(key="chat_messages")
if "steps" not in st.session_state:
    st.session_state.steps = {}

for index, msg in enumerate(message_history.messages):
    with st.chat_message(msg.type):
        if str(index) in st.session_state.steps:
            for step in st.session_state.steps[str(index)]:
                with st.status(f"工具调用: {step[0].tool}", state="complete"):
                    st.write(step[1])
        st.write(msg.content)

# ================== 核心对话逻辑 =======================
prompt_input = st.chat_input(placeholder='请提问...')

if prompt_input:
    if not api_key:
        st.info('请输入 API Key')
        st.stop()
    
    os.environ["DASHSCOPE_API_KEY"] = api_key
    st.chat_message('human').write(prompt_input)

    # 1. 准备组件
    llm = ChatTongyi(model_name=model_name, streaming=True, temperature=temperature)
    tools = [DuckDuckGoSearchRun(name="Search")]
    tool_desc = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])

    # 2. 构造最原始的 ReAct 模板
    template = """Answer the following questions. You have access to:
{tools}

Use this format:
Question: {input}
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation)
Thought: I now know the final answer
Final Answer: the final answer

Begin!
Question: {input}
Thought: {agent_scratchpad}"""

    prompt = ChatPromptTemplate.from_template(template)

    # 3. 【核心修复】不使用 assign，使用 RunnableLambda 纯手动处理输入
    # 这种方式直接避开了 Pydantic 对复杂 Runnable 结构的校验
    def transform_input(x):
        return {
            "input": x["input"],
            "agent_scratchpad": format_log_to_str(x["intermediate_steps"]),
            "tools": tool_desc,
            "tool_names": tool_names
        }

    # 组装链：处理输入 -> 填充模板 -> 传给模型 -> 解析输出
    agent_chain = RunnableLambda(transform_input) | prompt | llm | ReActSingleInputOutputParser()

    executor = AgentExecutor(
        agent=agent_chain,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    # 4. 运行
    with st.chat_message('ai'):
        st_cb = StreamlitCallbackHandler(st.container())
        try:
            # 直接输入字典
            response = executor.invoke(
                {"input": prompt_input},
                config={"callbacks": [st_cb]}
            )
            st.write(response['output'])
            new_index = str(len(message_history.messages) - 1)
            st.session_state.steps[new_index] = response['intermediate_steps']
        except Exception as e:
            st.error(f"还是报错了，这可能是环境深层冲突: {e}")
