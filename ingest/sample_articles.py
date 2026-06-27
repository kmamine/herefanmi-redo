"""Bundled sample articles from credible sources.

Used to seed SQLite + ChromaDB when live scraping is unavailable (CI, demos,
the docker-compose smoke test). Content is paraphrased, not scraped.
"""

from __future__ import annotations

from hrf_scraper.dedup import content_hash
from hrf_shared.contracts import Article

_RAW = [
    (
        "Influenza Vaccination",
        "cdc",
        "https://www.cdc.gov/flu/prevent/vaccinations.htm",
        "The CDC recommends an annual influenza vaccine for everyone aged six months "
        "and older, with rare exceptions. Vaccination is the most effective way to "
        "reduce the risk of flu illness, hospitalization, and death. Flu vaccines "
        "cause antibodies to develop in the body about two weeks after vaccination. "
        "Vaccination is especially important for people at higher risk of serious "
        "complications, including older adults, young children, pregnant people, and "
        "those with chronic health conditions such as asthma, diabetes, or heart disease.",
    ),
    (
        "Vitamin D and Bone Health",
        "nhs",
        "https://www.nhs.uk/conditions/vitamins-and-minerals/vitamin-d/",
        "Vitamin D helps regulate the amount of calcium and phosphate in the body. "
        "These nutrients are needed to keep bones, teeth, and muscles healthy. A lack "
        "of vitamin D can lead to bone deformities such as rickets in children and "
        "bone pain caused by osteomalacia in adults. Adults need 10 micrograms of "
        "vitamin D a day. Between October and early March, consider taking a daily "
        "supplement because sunlight is not strong enough for the body to make vitamin D.",
    ),
    (
        "High Blood Pressure",
        "medlineplus",
        "https://medlineplus.gov/highbloodpressure.html",
        "High blood pressure, also called hypertension, is blood pressure that is "
        "higher than normal. Blood pressure changes throughout the day based on "
        "activities. Having blood pressure measures consistently above normal may "
        "result in a diagnosis of high blood pressure. The higher your blood pressure "
        "levels, the more risk you have for other health problems, such as heart "
        "disease, heart attack, and stroke. Lifestyle changes and medicines can help "
        "control high blood pressure to lower the risk of these complications.",
    ),
    (
        "Statins for Cardiovascular Risk",
        "medpagetoday",
        "https://www.medpagetoday.com/cardiology/prevention/statins-overview",
        "Statins are a class of medicines used to lower cholesterol levels in the "
        "blood. A large body of randomized controlled trial evidence shows that "
        "statin therapy reduces the risk of major cardiovascular events, including "
        "heart attack and stroke, in people at elevated risk. For most eligible "
        "patients, the cardiovascular benefits of statins outweigh the risk of side "
        "effects. Clinicians weigh individual risk factors when deciding whether to "
        "start statin therapy.",
    ),
    (
        "Type 2 Diabetes",
        "webmd",
        "https://www.webmd.com/diabetes/type-2-diabetes",
        "Type 2 diabetes is a condition in which the body does not use insulin "
        "properly, a state known as insulin resistance. At first the pancreas makes "
        "extra insulin to compensate, but over time it cannot keep up, and blood "
        "sugar rises. Lifestyle changes such as a balanced diet, regular physical "
        "activity, and weight management are central to managing type 2 diabetes. "
        "Some people also need oral medicines or insulin to keep blood glucose within "
        "a healthy range and reduce the risk of complications.",
    ),
    (
        "Mediterranean Diet",
        "healthline",
        "https://www.healthline.com/nutrition/mediterranean-diet-meal-plan",
        "The Mediterranean diet emphasizes fruits, vegetables, whole grains, legumes, "
        "nuts, and olive oil, with moderate amounts of fish and poultry and limited "
        "red meat. Numerous studies have associated this eating pattern with a lower "
        "risk of heart disease, stroke, and type 2 diabetes, as well as improved "
        "longevity. The diet is rich in fiber, healthy fats, and antioxidants, and is "
        "widely recommended by health authorities as part of a heart-healthy lifestyle.",
    ),
]


def sample_articles() -> list[Article]:
    return [
        Article(
            title=title,
            content=content,
            url=url,
            source=source,
            content_hash=content_hash(content),
        )
        for (title, source, url, content) in _RAW
    ]
