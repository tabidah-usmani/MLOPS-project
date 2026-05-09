"""
load_test.py — Live traffic generator for Fake News Detector
Sends real POST requests to your running API so Prometheus scrapes the
metrics and Grafana dashboards show live FAKE/REAL counts, latency,
drift ratio, and request rate.

Usage:
    python load_test.py --probe               # test every article, show which ones predict FAKE
    python load_test.py                       # mixed mode, 100 requests
    python load_test.py --mode fake           # only confirmed-FAKE articles  (triggers drift)
    python load_test.py --mode real           # only REAL articles             (triggers drift)
    python load_test.py --mode mixed          # balanced traffic
    python load_test.py --mode burst          # spike then quiet
    python load_test.py --reset --mode mixed  # reset drift window first
"""

import requests
import time
import argparse
import random
import sys
from datetime import datetime

API_BASE = "http://localhost:5000"

# ─────────────────────────────────────────────────────────────────────────────
# FAKE articles
# Written to match real fake-news training corpora (WELFake / ISOT datasets):
# short punchy sentences, passive "sources say", all-caps keywords,
# political buzzwords, no named authors, excessive punctuation.
# ─────────────────────────────────────────────────────────────────────────────
FAKE_ARTICLES = [

    # ── Conspiracy / political ────────────────────────────────────────────────
    "obama caught red handed sending billions to iran in secret deal. sources inside the "
    "white house say the president personally approved the transfer to fund terrorist groups "
    "across the middle east. the mainstream media is silent because they are all complicit "
    "in the coverup. congress demands answers but investigators have been threatened. "
    "share this story before it gets deleted. the american people deserve to know the truth "
    "about what their government is doing behind closed doors with taxpayer money.",

    "hillary clinton exposed for running massive pay to play scheme at state department. "
    "wikileaks documents prove she sold government secrets to china and russia in exchange "
    "for clinton foundation donations worth hundreds of millions. fbi agents on the case "
    "have been reassigned or fired to protect her. multiple witnesses have died under "
    "mysterious circumstances before they could testify. the deep state is protecting her "
    "because she knows too much about the crimes of the global elite.",

    "donald trump was right all along. new evidence proves obama wiretapped trump tower "
    "during the 2016 election in the biggest political scandal in american history. "
    "intelligence officials say the operation was run directly from the oval office. "
    "fake news media refuses to cover this because it destroys their russia collusion hoax. "
    "three fbi agents have come forward as whistleblowers and their lives are now in danger. "
    "the truth is finally coming out and heads will roll in washington very soon.",

    "george soros funding violent antifa mobs to overthrow the united states government. "
    "leaked bank statements show transfers of 50 million dollars to radical left wing groups "
    "planning domestic terrorism in major american cities. democrat politicians are in on it. "
    "soros met secretly with nancy pelosi and chuck schumer last month to coordinate the "
    "coming revolution against president trump and the american people. patriots must "
    "stand up now or lose the country forever to these globalist traitors and criminals.",

    "joe biden family crime syndicate exposed by whistleblower with documents proving "
    "hunter biden was paid 10 million dollars by chinese communist party for access to the "
    "white house. joe biden knew everything and personally approved the deals according to "
    "sources with knowledge of the arrangement. the fbi has been sitting on this evidence "
    "for two years to protect the democrat party before the midterm elections. "
    "the corruption goes all the way to the top of the obama administration.",

    # ── Health / medical ──────────────────────────────────────────────────────
    "doctors dont want you to know this miracle cure for diabetes discovered by a retired "
    "physician in florida. by eating this one fruit every morning patients have reversed "
    "type 2 diabetes in just 30 days without medication. big pharma is spending millions "
    "to suppress this information because it would destroy their insulin profits. "
    "thousands of patients have already cured themselves using this simple natural remedy. "
    "your doctor will never tell you because they make money keeping you sick and dependent.",

    "cancer cure hidden by pharmaceutical companies for 40 years finally revealed. a natural "
    "compound found in apricot seeds called b17 destroys cancer cells on contact without "
    "harming healthy tissue. the fda banned this treatment in the 1970s after lobbying by "
    "drug companies who feared losing billions in chemotherapy revenue. a brave oncologist "
    "lost his medical license for successfully treating terminal patients with this cure. "
    "share this before big pharma pressures facebook to remove it permanently.",

    "cdc whistleblower admits flu vaccine causes alzheimers disease in elderly patients. "
    "a senior scientist at the centers for disease control has gone into hiding after leaking "
    "internal documents proving the annual flu shot accelerates cognitive decline. "
    "nursing homes have seen a 300 percent spike in dementia cases since mandatory "
    "vaccination programs began. the government knows and is covering it up to protect "
    "vaccine manufacturers from billions in liability lawsuits. warn your elderly relatives.",

    "tap water across america is being secretly fluoridated at double the legal limit "
    "according to a study by independent researchers in california. the added fluoride "
    "causes thyroid damage and calcification of the pineal gland which controls human "
    "consciousness and spiritual awareness. the government has been using fluoride since "
    "world war 2 as a mass medication program to make the population passive and obedient. "
    "install a reverse osmosis filter immediately to protect yourself and your family.",

    "5g towers being activated at night when no one is watching are causing massive bird "
    "die offs and unexplained nose bleeds in children near cell towers according to "
    "independent scientists who have been deplatformed for telling the truth. the radiation "
    "from 5g is 100 times more powerful than 4g and has never been tested for safety on "
    "humans. telecom companies bribed regulators to approve the technology. "
    "your children are being used as guinea pigs by the wireless industry for profit.",

    # ── Election / voting ─────────────────────────────────────────────────────
    "election fraud confirmed in six swing states by forensic audit team. dominion voting "
    "machines were connected to the internet on election night in direct violation of "
    "federal law and logs show vote tallies were changed remotely by operatives in "
    "frankfurt germany. hundreds of thousands of ballots were printed on non-security "
    "paper and smuggled into counting centers in the middle of the night. "
    "this is the biggest crime in american history and no one in power will do anything.",

    "millions of illegal votes cast in 2020 election by dead people and non citizens "
    "confirmed by data analysts who compared voter rolls to death records and census data. "
    "in one county in michigan more votes were cast than there are registered voters. "
    "the media says there is no evidence of fraud because they are part of the coverup. "
    "a federal judge has sealed the evidence to protect the biden administration. "
    "true the vote investigators have been harassed and jailed for exposing the truth.",

    # ── Celebrity / entertainment ─────────────────────────────────────────────
    "tom hanks arrested in greece and extradited to face charges of child trafficking "
    "according to sources close to the investigation. the hollywood star has been on an "
    "interpol watchlist for two years connected to an international pedophile ring "
    "operating through the entertainment industry. several a-list celebrities have been "
    "named in sealed indictments that are about to be unsealed by federal prosecutors. "
    "this is why he suddenly left the united states last year. the arrests are coming.",

    "oprah winfrey exposed as key recruiter for elite satanic cult operating among "
    "hollywood celebrities and democrat politicians. a former member of the group has "
    "given testimony to a grand jury in new york naming oprah as the person who introduced "
    "them to the secret society. the mainstream media is protecting her because she "
    "controls many journalists and news organizations through advertising relationships. "
    "the entertainment industry has been covering up these crimes for decades.",

    # ── Immigration / race ────────────────────────────────────────────────────
    "invasion at the border: biden administration secretly flying illegal immigrants to "
    "republican states in the middle of the night on unmarked government charter planes. "
    "sources at multiple airports have confirmed they are seeing planeloads of single "
    "military age males being released into communities without background checks. "
    "the plan is to import enough new democrat voters to make it impossible for "
    "republicans to ever win a national election again. this is demographic replacement.",

    "un migration pact gives george soros and globalist organizations the power to "
    "override american immigration law and force open borders on the united states. "
    "congress voted on this in secret without telling the american people what they "
    "were signing away. once implemented no president will be able to stop the flow "
    "of unlimited immigration from third world countries. this is the end of america "
    "as a sovereign nation if patriots dont rise up and stop it right now.",

    # ── Technology / surveillance ─────────────────────────────────────────────
    "facebook and google are recording all your private conversations through your "
    "smartphone microphone even when the apps are closed according to a former "
    "silicon valley engineer who has gone into witness protection. the data is sold "
    "to government intelligence agencies and used to build psychological profiles "
    "of every american citizen. senator mark warner has been briefed but refuses "
    "to act because he receives massive campaign donations from big tech companies.",

    "bill gates patent for coronavirus was filed in 2015 proving the pandemic was "
    "planned in advance by the world economic forum and global health organizations. "
    "event 201 in october 2019 was a rehearsal for the exact scenario that unfolded. "
    "the goal is to use vaccine passports to create a global digital id system "
    "that will allow governments to control every aspect of human life including "
    "where you can travel work or spend money. the great reset is already underway.",

    # ── Economy / financial ───────────────────────────────────────────────────
    "federal reserve printing trillions of dollars in secret to bail out foreign banks "
    "while american families struggle with inflation and rising prices. a congressman "
    "who demanded an audit of the federal reserve was found dead in his home two weeks "
    "later. the private banking cartel that controls the money supply has been stealing "
    "from american workers for over a century. end the fed before they destroy "
    "the dollar completely and plunge america into a great depression.",

    "china buying up american farmland near military bases as part of a coordinated "
    "intelligence operation to spy on us military installations and contaminate "
    "the food supply according to a classified pentagon report that was leaked online. "
    "the biden administration has approved dozens of these purchases and blocked "
    "state level efforts to stop them. chinese agents have already been caught "
    "attempting to steal soil samples from fields near three air force bases.",

    # ── Religion / culture war ────────────────────────────────────────────────
    "public schools secretly teaching children satanic rituals disguised as social "
    "emotional learning curriculum approved by the biden department of education. "
    "parents in virginia and florida have found materials in their childrens backpacks "
    "with pentagrams and instructions for contacting spirits. teachers who refused "
    "to use the curriculum were fired. this is part of a coordinated attack on "
    "christianity and traditional american family values by the radical left.",

    "woke corporations paying employees to attend critical race theory training sessions "
    "that teach white workers they are inherently racist and must confess their privilege "
    "before they can be promoted. employees who refuse to attend are being fired and "
    "their unemployment claims are being denied. this is illegal discrimination but "
    "the biden justice department refuses to investigate. share this to expose "
    "the anti-american indoctrination happening inside corporate america right now.",

    # ── Military / geopolitical ───────────────────────────────────────────────
    "ukraine biolabs funded by hunter biden and the pentagon contained dangerous "
    "pathogens that were being weaponized for deployment against russia according "
    "to documents captured by russian forces and presented to the united nations. "
    "victoria nuland admitted in senate testimony that biolab facilities exist in ukraine. "
    "the mainstream media immediately began covering this up because it contradicts "
    "the official narrative about why russia launched its military operation.",

    "nato secretly planning false flag attack to be blamed on russia to justify "
    "direct military intervention in ukraine according to leaked communications "
    "between pentagon officials and british intelligence. a staged chemical weapons "
    "incident is being planned for a civilian area and will be broadcast live on "
    "cnn to generate public support for world war three. russia has filed a formal "
    "complaint at the un security council but the media is not reporting it.",
]

# ─────────────────────────────────────────────────────────────────────────────
# REAL articles — clean journalism style, named sources, measured language
# ─────────────────────────────────────────────────────────────────────────────
REAL_ARTICLES = [
    "scientists at johns hopkins university have published findings showing that regular "
    "physical exercise of at least 150 minutes per week significantly reduces the risk of "
    "developing type 2 diabetes and cardiovascular disease. the study followed twelve "
    "thousand participants over a decade and controlled for diet genetics and income. "
    "results were published in the new england journal of medicine and represent one "
    "of the largest longitudinal health studies conducted in north america in recent years.",

    "the european space agency successfully launched its latest earth observation satellite "
    "from the kourou spaceport in french guiana on thursday. the satellite is part of the "
    "copernicus programme and will monitor sea surface temperatures and ice sheet thickness "
    "with unprecedented precision. scientists expect the data to significantly improve "
    "climate models over its seven year operational lifetime according to esa officials "
    "who confirmed stable orbit insertion shortly after launch.",

    "federal reserve officials held interest rates steady at their latest policy meeting "
    "citing continued uncertainty about inflation trends and labor market conditions. the "
    "decision was unanimous among the twelve voting members of the federal open market "
    "committee. chair jerome powell said policymakers would need several more months of "
    "economic data before considering rate adjustments. markets responded with modest gains "
    "as investors had largely anticipated the outcome.",

    "apple reported quarterly revenue of 94.9 billion dollars exceeding analyst expectations "
    "by three percent. iphone sales accounted for roughly half of total income while services "
    "revenue including the app store and apple music grew eighteen percent year over year. "
    "the company announced an expansion of its share buyback programme and a modest dividend "
    "increase. chief executive tim cook cited strong performance in india and southeast asia "
    "as key contributors to growth during the period.",

    "nasa engineers confirmed the james webb space telescope has completed its primary mirror "
    "alignment and is now fully operational. the telescope is positioned 1.5 million "
    "kilometers from earth and will observe the universe in infrared wavelengths allowing "
    "astronomers to study the formation of the earliest galaxies. initial test images "
    "exceeded expectations in resolution according to the mission science team at the "
    "goddard space flight center in maryland.",

    "the world health organization warned that drug resistant infections now kill "
    "approximately 700000 people annually worldwide and could reach ten million deaths "
    "per year by 2050 without action. the report recommends that governments restrict "
    "antibiotic use in livestock farming and implement stricter prescription requirements. "
    "several major pharmaceutical companies have announced renewed investment in antibiotic "
    "research following years of declining commercial interest in the sector.",

    "mit researchers have developed a sodium ion battery that could reduce the cost of "
    "energy storage for renewable power grids. the cells demonstrated comparable energy "
    "density to lithium ion batteries using materials fifty times more abundant in the "
    "earth crust. the team published findings in nature energy and filed manufacturing "
    "patents. clean energy investors have expressed interest in funding a commercial pilot "
    "programme to test the technology at scale over the next three years.",

    "the international monetary fund revised its global economic growth forecast to 3.2 "
    "percent for the current year citing stronger performance in emerging markets and "
    "resilient consumer spending. the imf warned that geopolitical tensions and elevated "
    "debt levels represent downside risks. managing director kristalina georgieva called "
    "on governments to rebuild fiscal buffers and invest in structural reforms to improve "
    "long term productivity and economic resilience.",

    "a new archaeological excavation in southern turkey has uncovered artifacts dating "
    "approximately nine thousand years showing evidence of early agricultural settlements. "
    "researchers from the university of chicago and istanbul university discovered ceramic "
    "vessels stone tools and remains of domesticated wheat and barley at the site. "
    "the findings will be analyzed using radiocarbon dating and submitted to nature "
    "archaeology for peer review by specialists in neolithic settlement patterns.",

    "the united kingdom recorded economic growth of 0.3 percent in the last quarter "
    "slightly above forecasts from the office for budget responsibility. the services "
    "sector drove most of the expansion with financial services and technology companies "
    "showing strong performance. manufacturing output remained flat amid supply chain "
    "disruptions affecting the automotive and electronics industries according to "
    "the office for national statistics.",

    "researchers have discovered a new species of deep sea fish in the pacific ocean "
    "at depths exceeding three thousand meters. the creature was captured by remotely "
    "operated vehicles during a research expedition funded by the national oceanic and "
    "atmospheric administration. the fish displays a distinctive bioluminescent pattern "
    "along its lateral line that researchers believe may be used for communication. "
    "specimens will be analyzed at the smithsonian institution before formal species description.",

    "the supreme court ruled six to three that states cannot impose additional "
    "qualifications for candidates seeking federal office beyond those listed in the "
    "constitution. the decision written by justice amy coney barrett reversed a state "
    "court ruling and was seen as a significant clarification of constitutional election law. "
    "legal scholars said the ruling would affect similar laws in several other states "
    "that had enacted comparable restrictions in recent years.",

    "boeing announced it would cut approximately 2000 jobs in its commercial aviation "
    "division as part of a restructuring plan following continued losses in the division. "
    "the company reported a quarterly loss of 1.4 billion dollars citing higher production "
    "costs and delivery delays on its 737 max and 787 dreamliner programmes. chief "
    "executive david calhoun said the company was taking difficult but necessary steps "
    "to return the business to profitability over the next two years.",

    "the biden administration announced a new 60 billion dollar aid package for ukraine "
    "including additional air defense systems artillery ammunition and armored vehicles. "
    "the package was approved by congress after months of debate and represents the largest "
    "single tranche of military assistance provided since russia invaded in february 2022. "
    "pentagon officials said deliveries would begin within weeks and would meaningfully "
    "strengthen ukrainian defensive capabilities along the eastern front.",

    "google announced it would invest 100 billion dollars in artificial intelligence "
    "infrastructure over the next five years including new data centers and custom "
    "semiconductor chips designed specifically for machine learning workloads. the "
    "investment represents a significant acceleration of the company existing ai strategy "
    "and comes as competition with microsoft and openai intensifies across enterprise "
    "and consumer markets according to chief executive sundar pichai.",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def ts():
    return datetime.now().strftime("%H:%M:%S")

def print_result(i, total, label, confidence, latency_ms, drift):
    filled = int(confidence * 20)
    bar    = "█" * filled + " " * (20 - filled)
    color  = "\033[91m" if label == "FAKE" else "\033[92m"
    drift_s = " \033[93m[DRIFT!]\033[0m" if drift else ""
    print(
        f"  [{ts()}] #{i:>3}/{total}  "
        f"{color}{label}\033[0m  "
        f"conf={confidence:.2f} [{bar}]  "
        f"{latency_ms:>6.1f}ms{drift_s}"
    )

def predict(text):
    try:
        r = requests.post(f"{API_BASE}/predict", json={"text": text}, timeout=10)
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        print(f"\n\033[91m  ERROR: Cannot reach {API_BASE}/predict\033[0m")
        print("  Make sure your Docker containers are running:")
        print("  docker compose -f docker/docker-compose.yml --project-directory . up -d\n")
        sys.exit(1)

def reset_drift():
    r = requests.post(f"{API_BASE}/reset", timeout=5)
    print(f"  [{ts()}] Drift window reset → {r.json()['status']}")

def check_drift():
    r   = requests.get(f"{API_BASE}/drift", timeout=5)
    d   = r.json()
    print(
        f"\n  Drift status : {d['status']}\n"
        f"  Window       : {d['window_size']}/{d['window_capacity']}  "
        f"FAKE={d['fake_ratio']:.0%}  REAL={d['real_ratio']:.0%}  "
        f"Threshold={d['threshold']:.0%}"
    )


# ─── Probe mode: find which articles the model actually calls FAKE ────────────

def run_probe():
    print(f"\n\033[1m  Mode: PROBE — testing every article individually\033[0m\n")
    confirmed_fake = []
    confirmed_real = []

    print("  ── FAKE candidates ──────────────────────────────────────────")
    for i, art in enumerate(FAKE_ARTICLES):
        data, status = predict(art)
        if status == 200 and 'label' in data:
            label      = data['label']
            confidence = data['confidence']
            marker     = "✓ FAKE" if label == "FAKE" else "✗ REAL"
            color      = "\033[91m" if label == "FAKE" else "\033[33m"
            print(f"  {color}[{marker}]\033[0m conf={confidence:.2f}  {art[:80].strip()}...")
            if label == "FAKE":
                confirmed_fake.append(art)
        time.sleep(0.2)

    print("\n  ── REAL candidates ──────────────────────────────────────────")
    for i, art in enumerate(REAL_ARTICLES):
        data, status = predict(art)
        if status == 200 and 'label' in data:
            label      = data['label']
            confidence = data['confidence']
            marker     = "✓ REAL" if label == "REAL" else "✗ FAKE"
            color      = "\033[92m" if label == "REAL" else "\033[33m"
            print(f"  {color}[{marker}]\033[0m conf={confidence:.2f}  {art[:80].strip()}...")
            if label == "REAL":
                confirmed_real.append(art)
        time.sleep(0.2)

    print(f"\n  ── Probe summary ────────────────────────────────────────────")
    print(f"  FAKE articles correctly predicted FAKE : {len(confirmed_fake)}/{len(FAKE_ARTICLES)}")
    print(f"  REAL articles correctly predicted REAL : {len(confirmed_real)}/{len(REAL_ARTICLES)}")
    print(f"\n  Run  python load_test.py --mode mixed  to send live traffic.\n")


# ─── Traffic modes ────────────────────────────────────────────────────────────

def run_mixed(count, delay):
    """Strict 50/50 interleave of FAKE and REAL."""
    print(f"\n\033[1m  Mode: MIXED  ({count} requests, {delay}s delay)\033[0m\n")
    pool = []
    half = count // 2
    pool += [('F', a) for a in random.choices(FAKE_ARTICLES, k=half)]
    pool += [('R', a) for a in random.choices(REAL_ARTICLES, k=count - half)]
    random.shuffle(pool)
    for i, (_, article) in enumerate(pool, 1):
        data, status = predict(article)
        if status == 200 and 'label' in data:
            print_result(i, count, data['label'], data['confidence'],
                         data['latency_ms'], data['drift_detected'])
        time.sleep(delay)

def run_fake_flood(count, delay):
    print(f"\n\033[1m  Mode: FAKE FLOOD  ({count} requests) — drift alert expected after 50\033[0m\n")
    for i in range(1, count + 1):
        data, status = predict(random.choice(FAKE_ARTICLES))
        if status == 200 and 'label' in data:
            print_result(i, count, data['label'], data['confidence'],
                         data['latency_ms'], data['drift_detected'])
            if data['drift_detected'] and i % 10 == 0:
                check_drift()
        time.sleep(delay)

def run_real_flood(count, delay):
    print(f"\n\033[1m  Mode: REAL FLOOD  ({count} requests) — drift alert expected after 50\033[0m\n")
    for i in range(1, count + 1):
        data, status = predict(random.choice(REAL_ARTICLES))
        if status == 200 and 'label' in data:
            print_result(i, count, data['label'], data['confidence'],
                         data['latency_ms'], data['drift_detected'])
        time.sleep(delay)

def run_burst(delay):
    print(f"\n\033[1m  Mode: BURST  (30 fast → 10s pause → 10 slow)\033[0m\n")
    all_articles = FAKE_ARTICLES + REAL_ARTICLES
    print("  Phase 1: rapid burst (30 requests, 0.1s delay)")
    for i in range(1, 31):
        data, status = predict(random.choice(all_articles))
        if status == 200 and 'label' in data:
            print_result(i, 40, data['label'], data['confidence'],
                         data['latency_ms'], data['drift_detected'])
        time.sleep(0.1)
    print("\n  Phase 2: quiet (10s pause — watch request rate drop on Grafana)")
    time.sleep(10)
    print("\n  Phase 3: slow trickle (10 requests, 2s delay)")
    for i in range(31, 41):
        data, status = predict(random.choice(all_articles))
        if status == 200 and 'label' in data:
            print_result(i, 40, data['label'], data['confidence'],
                         data['latency_ms'], data['drift_detected'])
        time.sleep(2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  choices=['mixed','fake','real','burst'], default='mixed')
    parser.add_argument('--count', type=int,   default=100)
    parser.add_argument('--delay', type=float, default=0.5)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--probe', action='store_true',
                        help='Test every article individually to see which ones predict FAKE')
    args = parser.parse_args()

    print(f"\n\033[1mFake News Detector — Live Traffic Generator\033[0m")
    print(f"  API:     {API_BASE}")
    print(f"  Grafana: http://localhost:3000")
    print(f"  Metrics: {API_BASE}/metrics\n")

    try:
        h = requests.get(f"{API_BASE}/health", timeout=5).json()
        print(f"  API status: \033[92m{h['status']}\033[0m  version={h['version']}")
    except Exception:
        print(f"\033[91m  API not reachable at {API_BASE}\033[0m")
        sys.exit(1)

    if args.probe:
        run_probe()
        return

    if args.reset:
        reset_drift()

    if   args.mode == 'mixed': run_mixed(args.count, args.delay)
    elif args.mode == 'fake':  run_fake_flood(args.count, args.delay)
    elif args.mode == 'real':  run_real_flood(args.count, args.delay)
    elif args.mode == 'burst': run_burst(args.delay)

    print("\n")
    check_drift()
    print(f"\n  Done. Open Grafana at http://localhost:3000 to see the metrics.\n")


if __name__ == '__main__':
    main()