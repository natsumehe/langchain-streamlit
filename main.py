import streamlit as st
from langchain_community.chat_models import ChatTongyi
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# ================== 页面配置 =======================
st.set_page_config(page_title='基于 Streamlit 的千问聊天机器人', layout='wide')
st.title("🤖 Qwen + LangChain 智能助手")

# ================== 左边栏配置部分 =======================
with st.sidebar:
    st.header("配置参数")
    api_key = st.text_input('Alibaba DashScope API Key', type='password')
    model = st.selectbox('选择模型', ('qwen-max', 'qwen-plus', 'qwen-turbo'))
    temperature = st.slider('Temperature', 0.0, 2.0, value=0.6, step=0.1)
    
    if st.button('清空聊天历史记录'):
        st.session_state.clear()
        st.rerun()

# ================== 核心逻辑初始化 =======================
# 1. 初始化消息记录
message_history = StreamlitChatMessageHistory(key="chat_messages")

# 2. 初始化用于存储中间思考步骤的状态
if "steps" not in st.session_state:
    st.session_state.steps = {}

# 3. 默认欢迎语
if len(message_history.messages) == 0:
    message_history.add_ai_message('你好！我是基于通义千问的助手，有什么可以帮你的吗？')

# 渲染历史消息
for index, msg in enumerate(message_history.messages):
    with st.chat_message(msg.type):
        # 渲染该消息对应的工具调用步骤（如果有）
        if str(index) in st.session_state.steps:
            for step in st.session_state.steps[str(index)]:
                with st.status(f"工具调用: {step[0].tool}", state="complete"):
                    st.write(f"输入: {step[0].tool_input}")
                    st.write(step[1])
        st.write(msg.content)

# ================== 聊天输入与逻辑 =======================
prompt = st.chat_input(placeholder='请提问，例如：现在巴黎的天气如何？')

if prompt:
    if not api_key:
        st.info('请在左侧输入 API Key 以开始对话')
        st.stop()

    # 展示用户输入
    st.chat_message('human').write(prompt)

    # 4. 构建 LLM 与工具
    llm = ChatTongyi(
        model_name=model,
        api_key=api_key,
        streaming=True,
        temperature=temperature,
    )
    
    tools = [DuckDuckGoSearchRun(name='Search')]

    # 5. 构建 Prompt Template (必须包含 chat_history 和 agent_scratchpad)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的 AI 助手。你可以通过搜索工具获取实时信息。"),
        MessagesPlaceholder(variable_name="chat_history"), # 历史上下文
        ("human", "{input}"),                             # 当前用户输入
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Agent 思考占位符
    ])

    # 6. 初始化 Memory (Key 必须与 Prompt 中的变量名对应)
    memory = ConversationBufferWindowMemory(
        chat_memory=message_history,
        return_messages=True,
        memory_key='chat_history',
        output_key='output',
        k=5
    )

    # 7. 创建 Agent 和 Executor
    agent = create_openai_functions_agent(llm, tools, prompt_template)
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True
    )

    # 8. 执行并展示 AI 回复
    with st.chat_message('ai'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = executor.invoke(
            {"input": prompt}, 
            config={"callbacks": [st_cb]}
        )
        
        answer = response['output']
        st.write(answer)

        # 保存中间步骤以便在页面刷新后依然能显示
        # 注意：这里减 1 是因为 invoke 结束后 message_history 已经增加了新的 AI 消息
        new_index = str(len(message_history.messages) - 1)
        st.session_state.steps[new_index] = response['intermediate_steps']