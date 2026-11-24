# Trendora AI: Your Personal Content Manager

Trendora AI არის AI აგენტი კონტენტ კრეატორებისა და მცირე ბიზნესებისთვის. ის ეხმარება იდეების შენახვასა და ორგანიზებაში, კონტენტ-კალენდრის შექმნაში, სცენარებისა და პოსტების წერაში, ტრენდების მოძიებასა და გამოქვეყნების საუკეთესო დროის შერჩევაში. აგენტი ქართულად საუბრობს და წაშლამდე მომხმარებლისგან დადასტურებას ითხოვს.

აგებულია Python-ზე, **LangGraph StateGraph**-ისა და **Groq** (`openai/gpt-oss-120b`) მოდელის გამოყენებით.

## Tools

| Tool | დანიშნულება |
|---|---|
| `save_business_profile` | ბიზნესისა და სამიზნე აუდიტორიის შენახვა (პერსონალიზაციისთვის) |
| `save_idea`, `list_ideas`, `delete_idea` | კონტენტ იდეების შენახვა, ჩვენება და წაშლა (წაშლა დადასტურებით) |
| `create_content_calendar` | კვირისა და თვის კონტენტ-კალენდარი |
| `write_content` | Instagram პოსტები, Reels და TikTok სცენარები |
| `get_trends` | ტრენდები (Google Trends RSS) და მარკეტინგული სიახლეები (Google News RSS) |
| `repurpose_idea` | ერთი იდეის ადაპტირება სხვადასხვა პლატფორმისთვის |
| `best_posting_time` | გამოქვეყნების/რეკლამის სავარაუდო საუკეთესო დრო |
| `hashtags_and_ad_copy` | ჰეშთეგები და სარეკლამო ტექსტები |
| `personalized_recommendations` | რეკომენდაციები ბიზნესის სფეროსა და აუდიტორიის მიხედვით |
| `get_weather` | ამინდი (Open-Meteo): ამინდის მიხედვით კონტენტის იდეებისთვის |
| `search_books` | წიგნების ძიება (Open Library) მარკეტინგის თემებზე |

## Memory და Approval Flow

- **Memory:** `MemorySaver` checkpointer და `thread_id`. ერთ საუბარში აგენტს წინა შეტყობინებები ახსოვს.
- **Approval Flow:** `delete_idea`-ის წინ გრაფი `interrupt()`-ზე ჩერდება და აჩვენებს, რის წაშლას აპირებს. `Command(resume=True)` ასრულებს წაშლას, `Command(resume=False)` ტოვებს მონაცემს უცვლელს.

## ფაილები

| ფაილი | დანიშნულება |
|---|---|
| `agent.ipynb` | Colab-ის ნამუშევარი, ტესტები და შეფასების ცხრილი |
| `agent.py` | აგენტის საერთო კოდი (`build_graph()`) |
| `app.py` | Streamlit ჩატი |
| `main.py` | Terminal-ის ვერსია |
| `requirements.txt` | საჭირო ბიბლიოთეკები |

## გაშვება

1. აიღე Groq API გასაღები: https://console.groq.com/keys
2. დააყენე ბიბლიოთეკები: `pip install -r requirements.txt`
3. დააყენე გასაღები გარემოს ცვლადად:
   - Linux/Mac: `export GROQ_API_KEY="..."`
   - Windows (PowerShell): `$env:GROQ_API_KEY="..."`
4. Terminal: `python main.py`
5. Streamlit: `streamlit run app.py`
   (გასაღები შეგიძლია `.streamlit/secrets.toml`-შიც ჩაწერო: `GROQ_API_KEY = "..."`)

Colab-ში გასაღები ინახება **Secrets**-ში სახელით `GROQ_API_KEY`.

## ტესტები

ტესტების შედეგები და შეფასების ცხრილი იხილე `agent.ipynb`-ში: დამატება, მეხსიერება, ამინდი, წიგნები, წაშლა დადასტურებით და უარით.

## შეზღუდვები

- Groq-ის უფასო გეგმას აქვს ტოკენების ლიმიტი (წუთში 8000 ამ მოდელზე), ამიტომ მოდელს ბოლო 10 შეტყობინებას ვუგზავნით და დიდ პასუხებს შორის პაუზა შეიძლება დაგჭირდეს.
- ხანდახან მოდელი გრძელ ქართულ პასუხზე იმეორებს სიტყვას; ამიტომ პასუხის სიგრძე შეზღუდულია.
- საუბრის მეხსიერება დროებითია (`MemorySaver`), ახალ სესიაში იკარგება. იდეები ინახება ლოკალურ JSON ფაილში (`trendora_data.json`).
- გამოქვეყნების დროის რეკომენდაცია სავარაუდოა და საკუთარი Insights-ით უნდა გადაამოწმო.
- ტრენდები დამოკიდებულია Google RSS ხელმისაწვდომობაზე.

## უსაფრთხოება

API გასაღები არ ინახება კოდში. `.env`, `.streamlit/secrets.toml` და `trendora_data.json` `.gitignore`-შია.
