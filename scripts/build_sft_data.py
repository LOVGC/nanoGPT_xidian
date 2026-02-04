from __future__ import annotations

import json
import random
from pathlib import Path


def build_dataset() -> list[dict[str, str]]:
    data: list[dict[str, str]] = []

    def add(instruction: str, input_text: str, output_text: str) -> None:
        data.append(
            {
                "instruction": instruction.strip(),
                "input": input_text.strip(),
                "output": output_text.strip(),
            }
        )

    translation_cn_en = [
        ("你好", "Hello."),
        ("早上好", "Good morning."),
        ("今天天气很好", "The weather is nice today."),
        ("我喜欢阅读", "I like reading."),
        ("请关门", "Please close the door."),
        ("这本书很有趣", "This book is very interesting."),
        ("我们走吧", "Let's go."),
        ("他正在做饭", "He is cooking."),
        ("我迷路了", "I am lost."),
        ("祝你生日快乐", "Happy birthday to you."),
        ("我需要帮助", "I need help."),
        ("这里有空位吗", "Is there a seat here?"),
        ("我忘记了密码", "I forgot the password."),
        ("现在几点了", "What time is it now?"),
        ("我们明天见", "See you tomorrow."),
        ("这家店几点开门", "What time does this shop open?"),
        ("请稍等", "Please wait a moment."),
        ("我想喝水", "I want to drink water."),
        ("这个问题很简单", "This question is very simple."),
        ("我正在学习英语", "I am learning English."),
        ("请把灯关掉", "Please turn off the light."),
        ("我不太明白", "I don't quite understand."),
        ("我们需要一个计划", "We need a plan."),
        ("路上小心", "Be careful on the road."),
        ("这件衣服太大了", "This piece of clothing is too big."),
        ("我可以试一下吗", "Can I try it on?"),
        ("这里禁止吸烟", "Smoking is not allowed here."),
        ("会议推迟到下午", "The meeting is postponed to the afternoon."),
        ("我们已经到了", "We have arrived."),
        ("请帮我拍张照片", "Please take a photo for me."),
    ]
    for cn, en in translation_cn_en:
        add("翻译成英文", cn, en)

    translation_en_cn = [
        ("Thank you for your help.", "谢谢你的帮助。"),
        ("Where is the nearest station?", "最近的车站在哪里？"),
        ("I don't know.", "我不知道。"),
        ("The food is delicious.", "这顿饭很好吃。"),
        ("Please speak slowly.", "请说慢一点。"),
        ("I will be late.", "我会迟到。"),
        ("Can you repeat that?", "你能再说一遍吗？"),
        ("It is raining outside.", "外面在下雨。"),
        ("I need a taxi.", "我需要一辆出租车。"),
        ("What is your name?", "你叫什么名字？"),
        ("This is my friend.", "这是我的朋友。"),
        ("I am looking for a hotel.", "我在找一家酒店。"),
        ("The price is too high.", "价格太高了。"),
        ("I agree with you.", "我同意你的看法。"),
        ("See you next week.", "下周见。"),
        ("The meeting starts at nine.", "会议九点开始。"),
        ("Please write it down.", "请写下来。"),
        ("I have a question.", "我有一个问题。"),
        ("The movie was boring.", "这部电影很无聊。"),
        ("I want to learn Chinese.", "我想学中文。"),
        ("The train is arriving.", "火车正在进站。"),
        ("Please open the window.", "请打开窗户。"),
        ("I am very tired.", "我很累。"),
        ("We are ready to go.", "我们准备出发了。"),
        ("The room is clean.", "房间很干净。"),
        ("I made a mistake.", "我犯了一个错误。"),
        ("It works now.", "现在可以用了。"),
        ("Keep the change.", "不用找零。"),
        ("The internet is slow.", "网络很慢。"),
        ("Have a nice day.", "祝你有美好的一天。"),
    ]
    for en, cn in translation_en_cn:
        add("翻译成中文", en, cn)

    summaries = [
        (
            "Tom lost his keys on the way home. After searching his bag, he found them in his jacket pocket.",
            "Tom found his lost keys in his jacket pocket.",
        ),
        (
            "The class planted seeds in small pots. They watered them every day and tiny sprouts appeared a week later.",
            "The seeds sprouted after a week of watering.",
        ),
        (
            "The train was delayed by heavy rain. Passengers waited patiently and the train finally departed an hour later.",
            "Heavy rain delayed the train by an hour.",
        ),
        (
            "Mia baked cookies for her neighbors. She wrapped them in small bags and delivered them after school.",
            "Mia baked and delivered cookies to her neighbors.",
        ),
        (
            "A boy saved his allowance for months. He bought the book he wanted and felt proud of his patience.",
            "He saved money and bought his book.",
        ),
        (
            "The library announced a renovation. It moved to a temporary room while workers updated the main hall.",
            "The library moved temporarily during renovations.",
        ),
        (
            "A dog learned to fetch a ball. Each time it returned the ball, the family cheered and played again.",
            "The dog learned to fetch and play with the family.",
        ),
        (
            "Volunteers cleaned the park on Saturday. They picked up trash and planted new flowers.",
            "Volunteers cleaned the park and planted flowers.",
        ),
        (
            "Lena practiced piano daily for a month. At the recital, she played confidently and received applause.",
            "Lena practiced and played well at her recital.",
        ),
        (
            "A power outage hit the town at night. People used candles and the electricity returned by morning.",
            "The town lost power overnight and it returned by morning.",
        ),
        (
            "小明去医院看望奶奶，带了水果和鲜花。奶奶很开心，他们一起聊天。",
            "小明去医院探望奶奶并陪她聊天。",
        ),
        (
            "学校举办运动会，同学们参加跑步和跳远。大家为彼此加油。",
            "学校运动会让大家参加项目并互相鼓励。",
        ),
        (
            "公司开会讨论新计划，大家提出了不同的建议，最后确定了目标。",
            "公司开会讨论计划并确定目标。",
        ),
        (
            "孩子们在雨后看见彩虹，开心地拍照并分享给朋友。",
            "雨后彩虹让孩子们开心拍照分享。",
        ),
        (
            "图书馆新增了很多科普书，学生们借阅后表示很有收获。",
            "图书馆新增科普书，学生收获很大。",
        ),
        (
            "A customer returned a broken item with a receipt. The store apologized and issued a refund.",
            "The store refunded a customer for a broken item.",
        ),
        (
            "The river overflowed after a storm. Volunteers filled sandbags and helped residents move valuables.",
            "Volunteers helped during a river flood.",
        ),
        (
            "Emma fell while learning to ride a bike. She tried again and eventually rode without support.",
            "Emma kept practicing and learned to ride a bike.",
        ),
        (
            "The cafe introduced a new menu. Customers tried the dishes and left positive reviews.",
            "Customers liked the cafe's new menu.",
        ),
        (
            "Two friends argued over a misunderstanding. They talked it out and became close again.",
            "They resolved a misunderstanding and reconciled.",
        ),
    ]
    for text, summary in summaries:
        add("请总结下面的内容", text, summary)

    paraphrases = [
        (
            "The meeting was canceled because the room was unavailable.",
            "The meeting was called off since the room was not available.",
        ),
        (
            "She quickly finished her homework and went outside.",
            "She finished her homework fast and went outdoors.",
        ),
        (
            "The package arrived earlier than expected.",
            "The delivery showed up sooner than we thought.",
        ),
        (
            "He forgot to lock the door before leaving.",
            "He left without locking the door.",
        ),
        (
            "The instructions were clear and easy to follow.",
            "The directions were simple and understandable.",
        ),
        (
            "I cannot attend the event due to a conflict.",
            "I have a conflict and can't go to the event.",
        ),
        (
            "The soup tastes salty.",
            "The soup is too salty.",
        ),
        (
            "Please send the report by Friday.",
            "Kindly submit the report before Friday.",
        ),
        ("她对结果感到满意。", "她对结果很满意。"),
        ("天气突然变冷了。", "天气一下子变得很冷。"),
        ("我需要更多时间考虑。", "我还需要一点时间想一想。"),
        ("他们决定推迟旅行计划。", "他们选择把旅行计划往后延。"),
        ("这本书内容丰富。", "这本书的信息很充实。"),
        ("今天的会议很高效。", "今天的会议效率很高。"),
        ("We should double-check the numbers.", "Let's verify the numbers again."),
        ("The cat slept on the warm chair.", "The cat rested on the warm chair."),
        ("He asked for feedback on his draft.", "He requested feedback on his draft."),
        (
            "The city built a new park downtown.",
            "A new park was built downtown by the city.",
        ),
        (
            "I enjoyed the concert last night.",
            "I had a great time at last night's concert.",
        ),
        ("Please keep your voice down.", "Please speak more quietly."),
    ]
    for text, para in paraphrases:
        add("改写下面的句子，保持原意", text, para)

    sentiment_tasks = [
        ("I love this movie.", "正面"),
        ("This is the worst service I have received.", "负面"),
        ("The package arrived yesterday.", "中性"),
        ("The room is clean and comfortable.", "正面"),
        ("The food was cold and bland.", "负面"),
        ("The meeting is scheduled for Monday.", "中性"),
        ("I am very satisfied with the product.", "正面"),
        ("The app keeps crashing.", "负面"),
        ("我今天上班。", "中性"),
        ("这次体验非常糟糕。", "负面"),
    ]
    for text, label in sentiment_tasks:
        add("判断情感（正面/负面/中性）", text, label)

    topic_tasks = [
        ("The team won the championship after a close match.", "体育"),
        ("The stock market rose after the announcement.", "经济"),
        ("A new smartphone was released today.", "科技"),
        ("The actor announced a new movie.", "娱乐"),
        ("Doctors recommend regular exercise.", "健康"),
        ("The school updated its curriculum.", "教育"),
        ("The company reported higher profits.", "经济"),
        ("The tennis player won her third title.", "体育"),
        ("Researchers built a faster computer chip.", "科技"),
        ("A famous singer released a new album.", "娱乐"),
    ]
    for text, label in topic_tasks:
        add("判断话题类别（科技/体育/经济/娱乐/健康/教育）", text, label)

    intent_tasks = [
        ("请帮我修改收货地址。", "请求"),
        ("我的订单怎么还没到？", "询问"),
        ("你好吗？", "闲聊"),
        ("我对这个服务不满意。", "投诉"),
        ("请给我推荐一部电影。", "请求"),
        ("你们几点上班？", "询问"),
        ("聊天一下吧。", "闲聊"),
        ("快递员态度太差了。", "投诉"),
        ("能帮我查一下余额吗？", "请求"),
        ("这个产品保修多久？", "询问"),
    ]
    for text, label in intent_tasks:
        add("用户意图分类（询问/投诉/请求/闲聊）", text, label)

    qa_tasks = [
        ("地球绕太阳一周大约需要多久？", "大约一年。"),
        ("水在摄氏多少度沸腾？", "在标准大气压下是100摄氏度。"),
        ("What is the capital of France?", "Paris."),
        ("Who wrote Pride and Prejudice?", "Jane Austen."),
        ("What gas do plants absorb for photosynthesis?", "Carbon dioxide."),
        ("1公里等于多少米？", "1000米。"),
        ("光速大约是多少？", "约3×10^8米/秒。"),
        ("How many days are in a leap year?", "366 days."),
        ("为什么天空是蓝色的？", "因为大气对蓝光散射更强。"),
        ("What is H2O commonly known as?", "Water."),
        ("太阳从哪边升起？", "从东边升起。"),
        ("How many continents are there?", "Seven."),
        ("What is the largest planet in our solar system?", "Jupiter."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
        ("电流的单位是什么？", "安培。"),
        ("人民币的符号是什么？", "¥。"),
        ("What is the boiling point of water in Fahrenheit?", "212°F."),
        ("香蕉富含哪种元素？", "钾。"),
        ("What is the square root of 81?", "9."),
        ("长城主要位于哪个国家？", "中国。"),
        ("Which ocean is the largest?", "The Pacific Ocean."),
        ("What is the chemical symbol for gold?", "Au."),
        ("一打有多少个？", "12个。"),
        ("What is the freezing point of water in Celsius?", "0°C."),
        ("月球绕地球一周大约多久？", "大约27.3天。"),
        ("What is the tallest mammal?", "The giraffe."),
        ("电话是谁发明的？", "亚历山大·贝尔。"),
        ("How many hours are in a day?", "24 hours."),
        ("What does CPU stand for?", "Central Processing Unit."),
        ("企鹅是鸟类吗？", "是的，但不能飞。"),
    ]
    for question, answer in qa_tasks:
        add("回答问题", question, answer)

    advice_topics = [
        ("准备面试", ["列出自己的亮点", "准备常见问题", "提前练习表达"]),
        ("提高英语听力", ["每天听短音频", "跟读模仿发音", "记录生词"]),
        ("养成早起习惯", ["固定睡眠时间", "早上少看手机", "准备简单早餐"]),
        ("缓解紧张情绪", ["深呼吸", "提前准备", "把注意力放在任务上"]),
        ("高效复习", ["做计划", "分块学习", "定期回顾"]),
        ("保持专注", ["关闭干扰", "使用番茄钟", "休息时离开座位"]),
        ("改善沟通", ["先倾听", "表达清晰", "确认对方理解"]),
        ("减轻拖延", ["把任务拆小", "先做5分钟", "设置明确截止时间"]),
        ("安排旅行", ["确认行程", "准备证件", "留出缓冲时间"]),
        ("健康饮食", ["多蔬菜", "控制糖分", "规律进餐"]),
        ("节约开支", ["记录支出", "减少冲动消费", "设定预算"]),
        ("学习编程", ["从基础语法开始", "多写小项目", "持续练习"]),
        ("保护视力", ["每隔20分钟远眺", "保持正确坐姿", "光线充足"]),
        ("提高写作", ["多阅读", "先列提纲", "反复修改"]),
        ("准备考试", ["整理错题", "模拟练习", "保证睡眠"]),
        ("改善睡眠", ["睡前放松", "减少咖啡因", "保持规律作息"]),
        ("时间管理", ["列优先级", "使用日程表", "预留空档"]),
        ("建立自信", ["记录小成就", "积极自我对话", "不断尝试"]),
        ("提升演讲", ["明确要点", "控制语速", "练习眼神交流"]),
        ("学习新技能", ["设定目标", "每天小步推进", "持续复盘"]),
    ]
    for topic, tips in advice_topics:
        output = "可以尝试：" + "；".join(tips) + "。"
        add("给出建议", f"我需要关于{topic}的建议", output)

    dialogue_requests = [
        ("我今天心情不好。", "听起来你有点低落。想聊聊发生了什么吗？"),
        ("帮我安排一下今天的计划。", "可以先列出最重要的三件事，再按优先级安排时间。"),
        (
            "我忘记备份文件了怎么办？",
            "先看看是否有历史版本或回收站，再考虑使用恢复工具。",
        ),
        ("我该怎么感谢同事的帮助？", "可以当面表达感谢，也可以发一封简短的致谢邮件。"),
        ("能推荐一部轻松的电影吗？", "可以试试轻松幽默的电影，比如温馨的家庭喜剧。"),
        ("我想开始跑步，有什么建议？", "从短距离开始，循序渐进，注意热身和拉伸。"),
        ("我和朋友吵架了。", "可以先冷静一下，再坦诚沟通彼此的感受。"),
        ("怎么提高专注力？", "减少干扰，设置固定的专注时间，并适当休息。"),
        ("我最近压力很大。", "可以尝试运动、规律作息，也可以找信任的人聊聊。"),
        ("请帮我写一段生日祝福。", "祝你生日快乐，愿你每一天都充满快乐与惊喜！"),
        ("能帮我改写这句话更正式吗？", "当然，可以把原句发给我，我来帮你改写。"),
        ("我要搬家了，有哪些注意事项？", "提前打包分类，更新地址信息，安排搬运时间。"),
        ("如何更快学习新软件？", "先看官方教程，跟着做小练习，再解决实际问题。"),
        ("我想省钱但不想太痛苦。", "从记录支出开始，减少小额冲动消费。"),
        ("我总是熬夜。", "试试固定睡前时间，减少睡前使用电子设备。"),
        ("我需要一个周末的放松计划。", "可以安排散步、阅读和轻松的兴趣活动。"),
        ("我想学做饭，先学什么？", "从简单菜开始，比如番茄炒蛋或炒青菜。"),
        ("我经常忘记带钥匙。", "可以在门口放一个提醒贴，或准备一个钥匙扣。"),
        ("我想提高口语表达。", "多练习复述，录音自检，并主动参加交流。"),
        ("我该怎么向老师请教问题？", "先把问题整理清楚，再说明你已尝试的方法。"),
        ("能帮我写个简短的自我介绍吗？", "当然，请告诉我你的背景和想突出的内容。"),
        ("我对未来有点迷茫。", "可以先设定短期目标，再逐步探索自己的兴趣方向。"),
        ("我做错了事，怎么道歉？", "真诚说明问题，表达歉意并提出改进措施。"),
        ("我想提升阅读速度。", "减少回读，先把握主旨，再阅读细节。"),
        ("我需要给客户发提醒。", "保持礼貌，明确提醒事项和时间。"),
        ("帮我想三个会议主题。", "可以考虑：目标回顾、问题梳理、行动计划。"),
        ("今天很忙，如何安排？", "先做最重要且最紧急的事项，再处理次要任务。"),
        ("请帮我写一个感谢短信。", "谢谢你的帮助，真的很感激，改天请你吃饭！"),
        ("我想培养一个新爱好。", "可以从你感兴趣的领域开始，比如摄影、烘焙或绘画。"),
        ("我担心明天的演讲。", "提前练习，准备提纲，放慢语速会更稳定。"),
        ("我想改善作息。", "每天固定睡醒时间，逐步提前入睡。"),
        ("我不知道该选哪门课。", "先看课程大纲和评价，再结合你的目标选择。"),
        ("我想写一份工作总结。", "按目标、过程、结果和下一步计划来写。"),
        ("帮我想一个周会开场白。", "大家好，我们先快速回顾上周进展，再安排本周重点。"),
        ("我需要一句话介绍产品。", "这是一款帮助你高效管理任务的轻量工具。"),
        ("我想给新同事发欢迎信息。", "欢迎加入团队，期待与你一起合作！"),
        ("我需要对客户解释延期原因。", "可以说明原因，给出新时间并表达歉意。"),
        ("帮我改成更礼貌的语气：马上给我回复。", "请方便时尽快回复，谢谢。"),
        ("我想知道如何开始写日记。", "每天写三句，记录今天的感受和一件小事。"),
        ("我害怕失败。", "把目标拆小，先完成可控的第一步。"),
        ("帮我列一个旅行清单。", "证件、充电器、衣物、洗漱用品和常用药。"),
        ("如何和室友相处更好？", "提前沟通规则，尊重彼此作息。"),
        ("我需要一个简单的晨间流程。", "起床喝水、拉伸5分钟、简单早餐。"),
        ("我想练习打字。", "每天练习10分钟，先保证准确再提高速度。"),
        ("帮我写一句会议提醒。", "提醒：今天下午3点有项目会议，请准时参加。"),
        ("我想学习摄影，有建议吗？", "从手机构图开始，多观察光线和角度。"),
        ("我觉得效率很低。", "先找出最耗时的环节，做小幅改进。"),
        ("请帮我润色一句话：请尽快处理。", "请尽快处理此事，感谢配合。"),
        ("我想准备一个团队活动。", "可以做简单投票，选择大家都能参加的时间和形式。"),
        ("我该怎么表达不同意见？", "先肯定对方，再提出你的观点和理由。"),
        ("我想给客户解释使用方法。", "先概述步骤，再提供一个简单示例。"),
        (
            "能推荐一本入门心理学的书吗？",
            "可以从通俗的心理学读物入手，比如基础入门书。",
        ),
        ("我想每天读书，但总是坚持不了。", "从每天10分钟开始，固定时间阅读。"),
        (
            "我需要一个简短的项目更新。",
            "项目已完成需求整理，正在推进开发，预计下周内测。",
        ),
        ("我想学会更好地倾听。", "少打断，对关键信息复述确认。"),
        ("我不知道如何开始一个对话。", "可以从天气、共同兴趣或最近的事情切入。"),
        ("我想写一段活动邀请。", "我们将在周六下午举办活动，欢迎你参加！"),
        ("我想给客户发送付款提醒。", "您好，温馨提醒账单已到期，如需协助请告知。"),
        ("我需要一句话总结会议。", "会议明确了目标和分工，下一步按计划推进。"),
        ("我想提高自律。", "设定清晰目标，记录进度并及时奖励自己。"),
        ("我有点拖延，怎么破？", "先完成最小可行的一步，建立节奏。"),
        ("我想写一个简短的离职说明。", "感谢公司的培养，我将于X月X日离职。"),
        ("帮我写一句反馈：讲解清楚。", "讲解清晰易懂，重点突出。"),
    ]
    for inp, out in dialogue_requests:
        add("作为友好的助手回答用户", inp, out)

    writing_tasks = [
        (
            "写一封简短的道歉邮件给同事，解释迟到原因。",
            "对不起今天迟到，早高峰交通堵塞导致我晚到。下次我会提前出门，避免影响工作。",
        ),
        (
            "写一个两句的生日祝福。",
            "祝你生日快乐，愿你的每一天都充满欢笑与惊喜。愿新的一岁心想事成！",
        ),
        (
            "写一个三句的小故事，主角是小猫。",
            "小猫在阳台上晒太阳。它看到一只蝴蝶，追着蝴蝶跑来跑去。最后它困了，窝在垫子里睡着了。",
        ),
        (
            "写一段简短的活动通知。",
            "本周五下午三点将在会议室举行分享活动，欢迎大家准时参加。请提前十分钟到场签到。",
        ),
        (
            "写一句产品宣传语。",
            "让任务管理更简单，让效率提升更自然。",
        ),
        (
            "写一段感谢客户的留言。",
            "感谢您对我们的支持与信任，我们会持续提供更好的服务。",
        ),
        (
            "写一个简短的自我介绍（两句）。",
            "大家好，我是李明，负责产品规划与用户研究。很高兴认识大家，期待一起合作。",
        ),
        (
            "写一个请假短信（两句）。",
            "我今天身体不适，需要请假一天。已安排好工作交接，感谢理解。",
        ),
        (
            "写一段鼓励的话，给正在备考的朋友。",
            "坚持到最后就是胜利，每一步努力都会有回报。相信自己，你可以的。",
        ),
        (
            "写一段周报开头（两句）。",
            "本周主要完成了需求梳理和原型设计。以下是具体进展与下周计划。",
        ),
        (
            "写一句委婉拒绝邀请的话。",
            "谢谢邀请，不过我那天已有安排，可能无法参加。",
        ),
        (
            "写一段简短的项目总结（两句）。",
            "项目按期完成并顺利上线，核心功能运行稳定。接下来将优化体验并持续迭代。",
        ),
        (
            "写一句提醒付款的短信。",
            "您好，温馨提醒账单已到期，如需帮助请随时联系。",
        ),
        (
            "写一段给新同事的欢迎语。",
            "欢迎加入团队，期待与你一起成长与合作！",
        ),
        (
            "写一个简短的会议总结标题。",
            "项目周会：进展回顾与下一步计划",
        ),
        (
            "写一句节日祝福（端午节）。",
            "端午安康，愿你平安喜乐。",
        ),
        (
            "写一段客户回访开场白。",
            "您好，我想了解一下您近期的使用体验，并看看是否需要我们提供帮助。",
        ),
        (
            "写一个简短的招聘岗位描述（两句）。",
            "我们正在招聘前端工程师，负责产品页面开发与优化。欢迎有相关经验的你加入。",
        ),
        (
            "写一句对团队的感谢。",
            "感谢团队的支持与配合，让项目顺利完成。",
        ),
        (
            "写一句鼓励儿童阅读的话。",
            "多读书会带你去很多有趣的地方。",
        ),
    ]
    for inst, out in writing_tasks:
        add("写作", inst, out)

    formatting_tasks = [
        ("从句子中提取日期", "会议安排在2024年5月3日进行。", "2024-05-03"),
        ("从句子中提取城市名", "我明天去上海出差，然后回北京。", "上海，北京"),
        ("把清单转成JSON数组", "苹果, 香蕉, 梨", '["苹果", "香蕉", "梨"]'),
        (
            "把下面内容改成项目符号",
            "待办：写报告；发邮件；开会",
            "- 写报告\n- 发邮件\n- 开会",
        ),
        ("提取句子中的电话号码", "请联系我：13800138000，有事再说。", "13800138000"),
        ("把颜色列表用英文逗号分隔", "红色 蓝色 绿色", "red, blue, green"),
        (
            "把下面字段整理成JSON",
            "name=Alice; age=23; city=Paris",
            '{"name":"Alice","age":23,"city":"Paris"}',
        ),
        ("提取邮箱地址", "请发到test@example.com，谢谢。", "test@example.com"),
        ("把这句话改成标题格式", "如何高效管理时间", "如何高效管理时间"),
        ("从句子中提取金额", "总价是￥128.50，已支付。", "128.50"),
        ("将以下内容转为CSV一行", "姓名:张三,年龄:20,城市:深圳", "张三,20,深圳"),
        ("提取产品型号", "型号：XQ-120B，颜色：黑色", "XQ-120B"),
        ("把括号内容提取出来", "这是一段文本（重要信息）请注意。", "重要信息"),
        ("把时间改成24小时制", "Meeting at 3:30 PM", "15:30"),
        ("从句子中提取国家名", "我想去日本旅行。", "日本"),
        ("把以下对话整理成Q/A", "Q:你是谁 A:我是助手", "Q: 你是谁\nA: 我是助手"),
        ("提取地址中的街道名", "地址：北京市朝阳区建国路88号", "建国路"),
        ("把数字列表排序", "3, 1, 2", "1, 2, 3"),
        ("提取句子中的颜色", "我买了红色的包。", "红色"),
        ("把英文月份缩写写全", "Jan", "January"),
    ]
    for inst, inp, out in formatting_tasks:
        add(inst, inp, out)

    reasoning_tasks = [
        ("比较大小", "12 和 19 哪个更大？", "19 更大。"),
        ("逻辑判断", "如果今天下雨就带伞。今天下雨了，我该做什么？", "带伞。"),
        ("常识推理", "冰放在室温下会发生什么？", "会融化成水。"),
        ("推断", "小王比小李高，小李比小张高，谁最高？", "小王最高。"),
        ("排序", "把 5、2、9 从小到大排列", "2，5，9。"),
        (
            "推理",
            "如果灯不亮可能是断电或灯泡坏了。先检查什么更合理？",
            "先检查是否断电。",
        ),
        ("判断", "一个偶数加上一个偶数结果是？", "偶数。"),
        ("推理", "今天是星期三，三天后是星期几？", "星期六。"),
        ("因果", "手机没电了，最可能的原因是什么？", "电池电量耗尽。"),
        ("选择", "下列哪个是水果：土豆/苹果/胡萝卜", "苹果。"),
        ("比较", "3/4 和 2/3 哪个更大？", "3/4 更大。"),
        ("判断", "如果一个数能被2整除，它是奇数还是偶数？", "偶数。"),
        ("推理", "如果商店关门了，你应该等到什么时候再去？", "等商店开门后。"),
        ("因果", "下雨后路面为什么会湿？", "因为雨水落在路面上。"),
        ("分类", "猫和狗属于什么动物类别？", "哺乳动物。"),
        ("比较", "100 和 99 哪个更大？", "100 更大。"),
        ("推理", "小明有5个苹果，吃掉2个，还剩几个？", "还剩3个。"),
        ("判断", "四边形有几条边？", "4条。"),
        ("推断", "夜晚天空更暗是因为太阳在哪里？", "太阳在地球另一侧。"),
        ("逻辑", "如果A是B的父亲，B是C的父亲，A是C的什么？", "A是C的祖父。"),
    ]
    for inst, inp, out in reasoning_tasks:
        add(inst, inp, out)

    for a in range(1, 21):
        for b in range(1, 6):
            add("计算下列表达式", f"{a} + {b}", str(a + b))

    for a in range(21, 41):
        for b in range(1, 6):
            add("计算下列表达式", f"{a} - {b}", str(a - b))

    for a in range(2, 12):
        for b in range(2, 12):
            add("计算下列表达式", f"{a} * {b}", str(a * b))

    rng = random.Random(42)
    rng.shuffle(data)

    if len(data) < 500:
        raise RuntimeError(f"dataset too small: {len(data)}")
    if len(data) > 1000:
        data = data[:1000]

    return data


def main() -> None:
    output_path = Path("data") / "sft_data.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(dataset)} records to {output_path}")


if __name__ == "__main__":
    main()
