from typing import Annotated  # 用于类型注解
from langgraph.graph import END, StateGraph, START  # 导入状态图的相关常量和类
from langgraph.graph.message import add_messages  # 用于在状态中处理消息
from langgraph.checkpoint.memory import MemorySaver  # 内存保存机制，用于保存检查点
from typing_extensions import TypedDict  # 用于定义带有键值对的字典类型
from langchain_ollama import ChatOllama  # 导入Ollama的聊天模型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入聊天提示模板和消息占位符
from langchain_core.messages import AIMessage, HumanMessage  # 导入AI和人类消息类


import functools  # 用于函数工具

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 使用注解确保消息列表使用add_messages方法处理

# 定义反思智能体的配置,实际应该从配置文件中读取
# 这里使用了一个简单的字典来模拟配置
reflection_agents = {
    "answer": {
        "name": "用户回答反思智能体",
        "description": "对生成的用户回答进行反思和反馈",
        "llm": ChatOllama,
        "model": "llama3.2",
        "max_tokens": 8192,
        "temperature": 0.2,
        "generation_system_message": 
            """
            You are a expertised assistant tasked with creating well-crafted, coherent, and engaging answers based on the user's input.
            Focus on clarity, structure, and quality to produce the best possible piece of answering.
            If the user provides feedback or suggestions, revise and improve the answer to better align with their expectations.
            """,
        "reflection_system_message": "You are a reviewer grading a answer submission. Provide critique and recommendations for the user's submission. Provide detailed recommendations, including requests for length, depth, style, etc.",
        "max_round": 2
    }
}    


# 输入状态，输出包含新生成消息的状态
def generation_node(generation, state: State) -> State:
    # 调用生成器(writer)，并将消息存储到新的状态中返回
    result = generation.invoke(state['messages'])
    return {"messages": [result]}

# 输入状态，输出带有反思反馈的状态
def reflection_node(reflect, state: State) -> State:
    # 创建一个消息类型映射，ai消息映射为HumanMessage，human消息映射为AIMessage
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    
    # 处理消息，保持用户的原始请求（第一个消息），转换其余消息的类型
    translated = [state['messages'][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state['messages'][1:]
    ]
    
    # 调用反思器(reflect)，将转换后的消息传入，获取反思结果
    res = reflect.invoke(translated)
    
    # 返回新的状态，其中包含反思后的消息
    return {"messages": [HumanMessage(content=res.content)]}

def create_agent(llm, system_message: str):
    """
    创建一个反思智能体，使用指定的LLM和系统消息。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    return prompt | llm


def create_agents(agent_config):
   
    if not agent_config:
        raise ValueError(f"Agent configuration is missing or invalid: {agent_config}")
    
    llm = agent_config["llm"](
        model=agent_config["model"],
        max_tokens=agent_config["max_tokens"],
        temperature=agent_config["temperature"]
    )

    generation_agent = create_agent(
        llm,
        agent_config["generation_system_message"]
    )

    # print("generation_node called with state:", generation_agent)

    reflection_agent = create_agent(
        llm,
        agent_config["reflection_system_message"]
    )
    return generation_agent, reflection_agent

def create_state_graph(generation_agent, reflection_agent, max_round=6):
    """
    创建一个状态图，定义生成和反思的流程。
    """
    g_node = functools.partial(generation_node, generation_agent)
    r_node = functools.partial(reflection_node, reflection_agent)

    # 定义条件函数，决定是否继续反思过程
    # 如果回合数达到最大值，则终止流程
    # 否则继续反思
    def should_continue(state: State):
        if len(state["messages"]) / 2 == max_round:
            return END  # 达到条件时，流程结束
        return "reflect"  # 否则继续进入反思节点

    builder = StateGraph(State)
    builder.add_node("writer", g_node)
    builder.add_node("reflect", r_node)
    builder.add_edge(START, "writer")

    # 在"writer"节点和"reflect"节点之间添加条件边
    # 判断是否需要继续反思，或者结束
    builder.add_conditional_edges("writer", should_continue)

    # 添加从"reflect"节点回到"writer"节点的边，进行反复的生成-反思循环
    builder.add_edge("reflect", "writer")

    # 创建内存保存机制，允许在流程中保存中间状态和检查点
    memory = MemorySaver()

    # 编译状态图，使用检查点机制
    return builder.compile(checkpointer=memory) 


def chat(input):
    """
    主函数，创建智能体并执行状态图。
    """
    agent_config = reflection_agents.get("answer")
    generation_agent, reflection_agent = create_agents(agent_config)
    graph = create_state_graph(generation_agent, reflection_agent, agent_config["max_round"])
    
    inputs = {
        "messages": [
            HumanMessage(content=input)
        ],
    }

    config = {"configurable": {"thread_id": "1"}}

    resp = graph.invoke(inputs, config)

    # 只保留第一条和最后一条消息
    if resp["messages"] and len(resp["messages"]) > 1:
        final_messages = [resp["messages"][0], resp["messages"][-1]]
    else:
        final_messages = resp["messages"]

    resp["messages"] = final_messages  # 更新响应中的消息

    return resp


if __name__ == "__main__":
    user_input = "请帮我写一篇关于人工智能的文章。"
    response = chat(user_input)

    print("\nFinal ChatHistory (only first and last message):")
    for msg in response["messages"]:
        # 判断消息类型并打印内容
        if isinstance(msg, HumanMessage):
            print(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"AI: {msg.content}")
        else:
            # 如果是其他类型的消息，直接打印内容    
            print(msg)
