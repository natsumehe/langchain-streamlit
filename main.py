# 导入必要的库
from langchain_community.chat_models import ChatTongyi # 1. 导入 ChatTongyi
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor # 2. 使用新的 Agent 创建方式
from langchain_core.prompts import MessagesPlaceholder # 3. 用于构建 Prompt
from langchain_core.prompts import ChatPromptTemplate # 4. 用于构建 Prompt
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory)

import streamlit as st

# 创建 浏览器 tab title
st.set_page_config(page_title='基于 Streamlit 的千问聊天机器人')

# ================== 左边栏配置部分 =======================
api_key = st.sidebar.text_input(
    'API Key', type='password'
)

model = st.sidebar.selectbox(
    'Model', ('qwen-max', 'qwen-plus', 'qwen-turbo')
)

temperature = st.sidebar.slider(
    'Temperature', 0.0, 2.0, value=0.6, step=0.1
)

# ================== 中间聊天部分 =======================
# 实例化用于存储聊天记录的 History 对象
message_history = StreamlitChatMessageHistory()
if not message_history.messages or st.sidebar.button('清空聊天历史记录'):
    message_history.clear()
    message_history.add_ai_message('有什么可以帮你的吗？')

    st.session_state.steps = {}

for index, msg in enumerate(message_history.messages):
    with st.chat_message(msg.type):
        for step in st.session_state.steps.get(str(index), []):
            if step[0].tool == '_Exception':
                continue
            with st.status(
                f'**{step[0].tool}**: {step[0].tool_input}',
                state='complete'
            ):
                st.write(step[0].log)
                st.write(step[1])

        st.write(msg.content)

prompt = st.chat_input(placeholder='请输入提问内容')
if prompt:
    if not api_key:
        st.info('请先输入 API Key')
        st.stop()

    st.chat_message('human').write(prompt)

    # 构建 Agent
    llm = ChatTongyi(
        model_name=model,
        api_key=api_key, # 5. 使用 api_key 参数
        streaming=True,
        temperature=temperature,
    )

    tools = [DuckDuckGoSearchRun(name='Search')]
    
    # 6. 构建新的 Agent (使用 OpenAI Functions style agent template)
    # 需要一个 ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Agent 的思考过程占位符
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt_template) # 7. 创建 Agent
    
    memory = ConversationBufferWindowMemory(
        chat_memory=message_history,
        return_messages=True,
        memory_key='chat_history',
        output_key='output',
        k=6
    )
    
    executor = AgentExecutor.from_agent_and_tools( # 8. 创建 Executor
        agent=agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    with st.chat_message('ai'):
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )
        response = executor({"input": prompt}, callbacks=[st_cb]) # 9. 调用方式更新为字典
        st.write(response['output']) # 10. 获取输出方式保持一致

        step_index = str(len(message_history.messages) - 1)
        st.session_state.steps[step_index] = response['intermediate_steps']