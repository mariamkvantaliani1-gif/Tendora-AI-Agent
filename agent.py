"""Trendora AI: ერთიანი აგენტის კოდი (Tools, State, StateGraph, Memory, Approval Flow).

ამ ფაილს იყენებს როგორც main.py (Terminal), ასევე app.py (Streamlit).
API გასაღები იკითხება GROQ_API_KEY გარემოს ცვლადიდან (ან Colab Secrets-იდან).
"""
import datetime
import json
import os
import xml.etree.ElementTree as ET
from typing import Annotated, Literal
from urllib.parse import quote

import requests
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

MODEL = "openai/gpt-oss-120b"
DB = os.environ.get("TRENDORA_DB", "trendora_data.json")


# ---------------------------------------------------------------- გასაღები
def ensure_api_key():
    """GROQ_API_KEY-ს ვეძებთ გარემოში; Colab-ში Secrets-იდან ვიღებთ."""
    if os.environ.get("GROQ_API_KEY"):
        return
    try:
        from google.colab import userdata
        os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
    except Exception as exc:
        raise RuntimeError("GROQ_API_KEY ვერ მოიძებნა. დააყენე გარემოს ცვლადად ან Streamlit Secrets-ში.") from exc


_llm_long = None


def get_llm_long():
    """მოდელი გენერაციის Tool-ებისთვის (კალენდარი, სცენარი და ა.შ.)."""
    global _llm_long
    if _llm_long is None:
        ensure_api_key()
        _llm_long = ChatGroq(model=MODEL, temperature=0.7, max_tokens=3000)
    return _llm_long


# ---------------------------------------------------------------- ინსტრუქციები
SYSTEM_KA = """შენ ხარ Trendora AI, პირადი კონტენტ მენეჯერი კონტენტ კრეატორებისა და მცირე ბიზნესებისთვის.
წერე მხოლოდ გამართულ, ბუნებრივ ქართულ ენაზე. ინგლისური სიტყვები გამოიყენე მხოლოდ პლატფორმის სახელებში
(Instagram, TikTok, Reels) და ჰეშთეგებში. პასუხი იყოს კონკრეტული, პრაქტიკული და გასაგები."""

AGENT_PROMPT = SYSTEM_KA + """

წესები:
- პასუხი იყოს მოკლე: მაქსიმუმ 6-8 წინადადება ან მოკლე სია. არასდროს გაიმეორო ერთი და იგივე სიტყვა ან ფრაზა.
- ბიზნესის აღწერისას შეინახე პროფილი save_business_profile Tool-ით.
- წაშლა: როცა მომხმარებელი იდეის წაშლას ითხოვს, დაუყოვნებლივ გამოიძახე delete_idea Tool (confirmed=false). დადასტურებას სისტემა თავად ითხოვს, შენ ტექსტით ნუ ჰკითხავ.
- როცა მომხმარებელი ამინდს ახსენებს ან გარე გადაღების იდეებს ითხოვს, გამოიყენე get_weather და შესთავაზე იდეები ამინდის მიხედვით (წვიმაზე შიდა გადაღებები, მზიან ამინდზე გარეთ).
- როცა მომხმარებელი სწავლას ან წიგნებს ახსენებს, გამოიყენე search_books მარკეტინგისა და კონტენტის თემებზე.
- ტრენდებზე პასუხი ყოველთვის get_trends Tool-ის შედეგებზე დააფუძნე.
- Tool-ის შედეგი გადმოეცი ქართულად, გამართულად და ადამიანურად."""


# ---------------------------------------------------------------- შენახვა (JSON ფაილი)
def _load():
    if os.path.exists(DB):
        with open(DB, encoding="utf-8") as f:
            return json.load(f)
    return {"ideas": [], "profile": {}}


def _save(data):
    with open(DB, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _profile_text():
    p = _load()["profile"]
    if not p:
        return "ბიზნესის პროფილი ჯერ არ არის შენახული."
    return f"ბიზნესი: {p.get('business')}; სამიზნე აუდიტორია: {p.get('audience')}; ტონი: {p.get('tone', '')}"


def ask_llm(task: str) -> str:
    msgs = [("system", SYSTEM_KA), ("human", f"{_profile_text()}\n\n{task}")]
    return get_llm_long().invoke(msgs).content


# ---------------------------------------------------------------- Tools
@tool
def save_business_profile(business: str, audience: str, tone: str = "") -> str:
    """Save the user's business field, target audience and preferred tone. Use this when the user describes their business."""
    data = _load()
    data["profile"] = {"business": business, "audience": audience, "tone": tone}
    _save(data)
    return "ბიზნესის პროფილი შენახულია."


@tool
def save_idea(title: str, details: str = "", category: str = "ზოგადი", platform: str = "ნებისმიერი") -> str:
    """Save a content idea with a category and platform so ideas stay organized."""
    data = _load()
    new_id = max([i["id"] for i in data["ideas"]], default=0) + 1
    data["ideas"].append({"id": new_id, "title": title, "details": details,
                          "category": category, "platform": platform,
                          "created": str(datetime.date.today())})
    _save(data)
    return f"იდეა შენახულია (ნომერი {new_id}): {title}"


@tool
def list_ideas(category: str = "") -> str:
    """List saved content ideas, optionally filtered by category."""
    ideas = _load()["ideas"]
    if category:
        ideas = [i for i in ideas if i["category"] == category]
    if not ideas:
        return "შენახული იდეები ვერ მოიძებნა."
    return "\n".join(f"#{i['id']} [{i['category']}/{i['platform']}] {i['title']} - {i['details']}" for i in ideas)


@tool
def delete_idea(idea_id: int, confirmed: bool = False) -> str:
    """Delete a saved idea by its number. Call with confirmed=false; the system itself asks the user for approval before deletion."""
    if not confirmed:
        return f"საჭიროა დადასტურება: ნამდვილად გსურთ #{idea_id} იდეის წაშლა?"
    data = _load()
    data["ideas"] = [i for i in data["ideas"] if i["id"] != idea_id]
    _save(data)
    return f"იდეა #{idea_id} წაიშალა."


@tool
def create_content_calendar(period: str, goal: str, platforms: str = "Instagram, TikTok") -> str:
    """Create a content calendar. period is 'კვირა' (7 days) or 'თვე' (30 days). goal is the business goal or theme."""
    days = 7 if period.startswith("კვირ") or period.lower().startswith("week") else 30
    start = datetime.date.today()
    dates = ", ".join(str(start + datetime.timedelta(d)) for d in range(days))
    return ask_llm(f"შექმენი {days} დღიანი კონტენტ-კალენდარი ცხრილის სახით (თარიღი | პლატფორმა | ფორმატი | თემა | მოკლე იდეა). "
                   f"თარიღები: {dates}. მიზანი: {goal}. პლატფორმები: {platforms}.")


@tool
def write_content(platform: str, content_format: str, idea: str) -> str:
    """Write an Instagram post, a Reels script or a TikTok script. content_format examples: 'პოსტი', 'Reels სცენარი', 'TikTok სცენარი'."""
    return ask_llm(f"დაწერე {platform} {content_format} იდეაზე: {idea}. "
                   "სცენარის შემთხვევაში მიუთითე: ყურადღების მისაპყრობი პირველი 3 წამი (hook), სცენები დროით, ტექსტი ეკრანზე, "
                   "ხმოვანი ტექსტი და მოწოდება მოქმედებისკენ (CTA).")


def _rss_titles(url, n=8):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    return [i.findtext("title") for i in root.iter("item")][:n]


@tool
def get_trends(topic: str = "social media marketing") -> str:
    """Get current trending searches in Georgia and latest marketing news about a topic."""
    parts = []
    try:
        t = _rss_titles("https://trends.google.com/trending/rss?geo=GE")
        parts.append("Google Trends (საქართველო): " + "; ".join(t))
    except Exception as e:
        parts.append(f"Google Trends ვერ ჩაიტვირთა: {e}")
    try:
        n = _rss_titles(f"https://news.google.com/rss/search?q={quote(topic)}&hl=en-US&gl=US&ceid=US:en")
        parts.append(f"სიახლეები თემაზე '{topic}': " + "; ".join(n))
    except Exception as e:
        parts.append(f"სიახლეები ვერ ჩაიტვირთა: {e}")
    return "\n".join(parts) + "\n(აგენტო, შეაჯამე ეს ქართულად და დაუკავშირე მომხმარებლის ბიზნესს.)"


@tool
def repurpose_idea(idea: str, platforms: str = "Instagram, TikTok, Facebook, LinkedIn") -> str:
    """Adapt one content idea to several social platforms (content repurposing)."""
    return ask_llm(f"ეს იდეა ადაპტირე თითოეული პლატფორმისთვის ცალ-ცალკე ფორმატით, სიგრძითა და ტონით: {idea}. პლატფორმები: {platforms}.")


@tool
def best_posting_time(platform: str, audience: str = "") -> str:
    """Suggest the likely best time to publish content or run ads on a platform (Georgia time, UTC+4)."""
    base = {
        "instagram": "სამ-ხუთ 11:00-13:00 და 19:00-21:00",
        "tiktok": "სამ-ხუთ 18:00-22:00, შაბათს 10:00-12:00",
        "facebook": "ოთხ-პარ 12:00-15:00",
    }.get(platform.lower(), "სამუშაო დღეებში 18:00-21:00")
    return ask_llm(f"{platform}-ზე გამოქვეყნების საუკეთესო დრო საქართველოს დროით. ზოგადი სავარაუდო დრო: {base}. "
                   f"აუდიტორია: {audience or 'იხილე პროფილი'}. აუხსენი მოკლედ და დაამატე, რომ ეს სავარაუდოა და საკუთარი Insights-ით უნდა გადაამოწმოს.")


@tool
def hashtags_and_ad_copy(subject: str, platform: str = "Instagram") -> str:
    """Generate hashtags and short advertising copy for a product, service or post."""
    return ask_llm(f"{platform}-ისთვის '{subject}'-ზე მომეცი: 15 ჰეშთეგი (ქართული და ინგლისური შერეული), 3 მოკლე სარეკლამო ტექსტი სხვადასხვა ტონით.")


@tool
def personalized_recommendations(question: str = "როგორ გავზარდო გაცნობიერება და გაყიდვები?") -> str:
    """Give personalized content and marketing recommendations based on the saved business profile."""
    return ask_llm(f"მომეცი 5 პერსონალიზებული, პრაქტიკული რეკომენდაცია ამ ბიზნესისთვის. კითხვა: {question}")


@tool
def get_weather(city: str = "Tbilisi") -> str:
    """Get current weather (temperature, precipitation, wind) for a city. Use it to suggest weather-based content ideas."""
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                       params={"name": city, "count": 1}, timeout=20)
    geo.raise_for_status()
    results = geo.json().get("results")
    if not results:
        return f"ქალაქი '{city}' ვერ მოიძებნა."
    place = results[0]
    w = requests.get("https://api.open-meteo.com/v1/forecast",
                     params={"latitude": place["latitude"], "longitude": place["longitude"],
                             "current": "temperature_2m,precipitation,wind_speed_10m"}, timeout=20)
    w.raise_for_status()
    cur = w.json()["current"]
    return (f"{place['name']}: ტემპერატურა {cur['temperature_2m']}°C, "
            f"ნალექი {cur['precipitation']} მმ, ქარი {cur['wind_speed_10m']} კმ/სთ")


@tool
def search_books(topic: str, limit: int = 3) -> str:
    """Search Open Library for books on a topic (for example social media marketing or storytelling). Returns a few titles and authors."""
    r = requests.get("https://openlibrary.org/search.json",
                     params={"q": topic, "limit": min(limit, 5),
                             "fields": "title,author_name,first_publish_year"}, timeout=20)
    r.raise_for_status()
    docs = r.json().get("docs", [])
    if not docs:
        return f"თემაზე '{topic}' წიგნები ვერ მოიძებნა."
    lines = []
    for d in docs:
        authors = ", ".join(d.get("author_name", ["უცნობი ავტორი"])[:2])
        lines.append(f"- {d.get('title')} ({authors}, {d.get('first_publish_year', '?')})")
    return "\n".join(lines)


ALL_TOOLS = [save_business_profile, save_idea, list_ideas, delete_idea, create_content_calendar,
             write_content, get_trends, repurpose_idea, best_posting_time,
             hashtags_and_ad_copy, personalized_recommendations, get_weather, search_books]


# ---------------------------------------------------------------- State და გრაფი
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def route_after_assistant(state: AgentState):
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None)
    if not calls:
        return END
    if any(c["name"] == "delete_idea" for c in calls):
        return "approval"
    return "tools"


def approval(state: AgentState) -> Command[Literal["tools", "assistant"]]:
    """წაშლამდე გრაფი ჩერდება (interrupt) და მომხმარებლისგან დადასტურებას ითხოვს."""
    last = state["messages"][-1]
    new_calls, rejected = [], False
    for c in last.tool_calls:
        if c["name"] == "delete_idea":
            idea_id = c["args"].get("idea_id")
            title = next((i["title"] for i in _load()["ideas"] if i["id"] == idea_id), "უცნობი იდეა")
            answer = interrupt({
                "action": "delete_idea",
                "idea_id": idea_id,
                "title": title,
                "message": f"აპირებს წაშალოს იდეა #{idea_id}: '{title}'. დაადასტურეთ?",
            })
            ok = answer is True or str(answer).strip().lower() in ("true", "დიახ", "კი", "yes", "y")
            if ok:
                c = {**c, "args": {**c["args"], "confirmed": True}}
            else:
                rejected = True
        new_calls.append(c)
    if rejected:
        msgs = [ToolMessage(content="მომხმარებელმა წაშლა უარყო, მონაცემი უცვლელია.", tool_call_id=c["id"])
                for c in last.tool_calls]
        return Command(goto="assistant", update={"messages": msgs})
    return Command(goto="tools", update={"messages": [AIMessage(content=last.content, tool_calls=new_calls, id=last.id)]})


def build_graph(checkpointer=None):
    """აგებს და აბრუნებს მზა გრაფს (მეხსიერებით)."""
    ensure_api_key()
    llm_with_tools = ChatGroq(model=MODEL, temperature=0.6, max_tokens=800).bind_tools(ALL_TOOLS)

    def assistant(state: AgentState):
        # მოდელს ვუგზავნით მხოლოდ ბოლო 10 შეტყობინებას (Groq-ის უფასო TPM ლიმიტის გამო)
        history = trim_messages(state["messages"], strategy="last", max_tokens=10,
                                token_counter=len, start_on="human", include_system=False)
        return {"messages": [llm_with_tools.invoke([SystemMessage(content=AGENT_PROMPT)] + history)]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("approval", approval)
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", route_after_assistant, ["tools", "approval", END])
    builder.add_edge("tools", "assistant")
    return builder.compile(checkpointer=checkpointer or MemorySaver())


def run_turn(graph, config, user_text=None, resume=None):
    """ერთი ნაბიჯი: ან ახალი შეტყობინება (user_text), ან დადასტურების პასუხი (resume=True/False).

    აბრუნებს {"type": "answer", "text": ...} ან {"type": "interrupt", "payload": {...}}.
    """
    if resume is not None:
        result = graph.invoke(Command(resume=resume), config)
    else:
        result = graph.invoke({"messages": [("user", user_text)]}, config)
    if result.get("__interrupt__"):
        return {"type": "interrupt", "payload": result["__interrupt__"][0].value}
    return {"type": "answer", "text": result["messages"][-1].content}
