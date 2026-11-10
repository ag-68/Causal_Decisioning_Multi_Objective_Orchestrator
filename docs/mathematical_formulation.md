# Mathematical Formulation: Constrained Causal Decisioning

A shop, a music app, or a delivery app can send a note. The note might be a buzz on the phone, an email in the inbox, or a short clip that plays while the person is already in the app. The note can bring a visit or a sale. The same note can make the person turn notes off.

**The problem.** If the only goal is today’s sales, the loudest note goes to almost everyone. People who would have come back anyway get a poke they did not need. People who hate being interrupted block the sender, and next week even a useful note cannot reach them. A friend who calls once with a good tip is welcome. The same friend calling five times gets blocked. The project is that everyday limit, written so a computer can respect it for a whole morning, and again the next morning. The same choice shows up for a song or a post. Once some other model has scored it, the decision is still one action under a budget: distribute it, show it less, hold it for review, or leave it alone. This page works that decision out as a note sent to a person.

**The approach.** The project does three things.

1. **Learn the extra a note causes.** For a person with a known shopping card and a known level of tiredness, estimate how much a particular note changes the chance of a buy, and how much it changes the chance of “stop,” compared with sending that person nothing.
2. **Choose one note per person, inside two house rules.** About 15 people in a thousand may say “stop.” About 300 phones in a thousand may buzz. Each person still gets their own note. The rules limit the morning’s totals.
3. **Carry tiredness and the charges into the next morning.** A buzz adds tiredness, and a quiet day lets it fade, so tomorrow’s extras are computed at a new tiredness. The opt-out charge and the channel charges are moved until this morning’s notes sit on the two house rules, and the next morning starts from those new charges. What is learned are the extra buy and the extra stop. The rule that spends them is one we wrote.

**How a training row is built.** The Hillstrom file has a shopping card for about 64,000 people. It records who they are and what they bought before. It does not record a tiredness, and it does not record a push or a clip inside the app. We keep each card and append two draws. One draw is a tiredness, chosen at random. The other is a message, chosen at random from silence, email, push, and audio. A formula we wrote then gives a buy chance from that card, that tiredness, and that message. A second formula gives a stop chance from that tiredness and that message. A coin flip turns each chance into a yes or a no. The stored row is the card, the tiredness, the message, and those two answers. The trees learn from the answers. The chance itself is not written on the row.

**What we expect.** On one morning, pricing the extra stops and the channel use should hold extra stops down and still keep the notes whose extra buy is worth the charge. Over the following mornings, the charges are tuned again each morning, starting from yesterday’s values, so extra stops stay near the allowance as tiredness changes. A scarce note should go to a person the note can still move. When the extra buy does not clear the charge, that person stays silent.

---

## 1. General problem

This section writes the shop problem with the names left open, so the same arithmetic can describe a message or any other one choice per person. Later sections fill the names in. The person is a customer. The choice is a message. The gain is a buy. The loss is someone saying “stop.”

One morning we have a list of \(N\) people. Person \(i\) has a shopping card \(x_{i,t}\) and a tiredness \(s_{i,t}\). We pick one message \(a\) from a short menu \(\mathcal{A}\). The menu always includes silence, written \(a_0\). Leaving the person alone is a real choice.

We only see what happened after the message we actually sent. Write \(Y_i(a)\) for the buy we would have recorded if person \(i\) had received message \(a\), and \(C_i(a)\) for the stop we would have recorded. The other messages stay what-ifs. The extra buy of message \(a\), compared with silence, is the average gap among people with this card and this tiredness:

\[
\tau_v(x, s, a) = \mathbb{E}\big[Y(a) - Y(a_0) \mid X = x,\, S = s\big],
\]

and the extra stop is the same kind of gap:

\[
\tau_c(x, s, a) = \mathbb{E}\big[C(a) - C(a_0) \mid X = x,\, S = s\big].
\]

The symbol \(\mathbb{E}\) means that average. If the message is silence, both gaps are 0. Ann might have a 70% chance of buying even if we send nothing, and a 72% chance after a push. Her extra buy is 2 percentage points. A ranking by the raw 72% would put her near the front. The gap of 2 points would not.

A rule \(\pi(a \mid x, s)\) picks one message for each person. In this project the pick is a single message, so \(\pi\) puts all of its weight on one \(a\). The morning’s job is to make the total extra buy as large as possible, while the average extra cost stays under a budget:

\[
\begin{aligned}
\max_{\pi} \quad
& \sum_{i=1}^{N} \sum_{a \in \mathcal{A}} \pi(a \mid x_i, s_i)\, \tau_v(x_i, s_i, a) \\
\text{s.t.} \quad
& \frac{1}{N} \sum_{i=1}^{N} \sum_{a \in \mathcal{A}} \pi(a \mid x_i, s_i)\, \tau_c^{(k)}(x_i, s_i, a) \le B_k,
  && k = 1, \ldots, K, \\
& \sum_{a \in \mathcal{A}} \pi(a \mid x_i, s_i) = 1, \quad \pi(a \mid x_i, s_i) \ge 0.
\end{aligned}
\]

Read the display from the top. The first line adds up, over people and messages, the extra buy of the message the rule picked. The second line says the average extra cost must stay at or under a budget \(B_k\). There can be several budgets. One is the opt-out allowance, for example 15 extra stops per 1,000 people. Another is a channel’s share, for example at most 300 phones buzzing. The last line says every person gets one message, and a share cannot be negative. Section 5 keeps those budgets by raising and lowering a price, called a charge, until the chosen notes sit inside them. The budget is the allowance. The charge is the price we move to land inside it.

Tiredness \(s_{i,t}\) is what makes the problem last more than one morning. Today’s message changes it, and tomorrow’s extra buy and extra stop are computed at the new tiredness. Section 6 writes that update down.

---

## 2. Assumptions

These are the promises the results rest on. Each one is a shop story of what we are relying on.

1. **The training message was picked by a coin.** On each stored row the message is silence, email, push, or audio, about one person in four for each. Every shopping card had the same chance of each message. A buy or a stop on that row can be traced to the message that was sent. Tiredness on that row was drawn by a coin too.
2. **Every message we might send was tried on some people.** Silence, email, push, and audio each went to about one person in four. The trees learn what a push does from the people who were sent a push. If no one had received a push, there would be no push rows to learn from. Some later scores also count a person more when their message was uncommon. The count is 1 divided by the share who got that message. One person in four means a count of 4. A share of zero would leave nothing to divide by. This promise is sometimes called overlap.
3. **Tiredness and “stop” are settings we wrote, and the trees see only yes or no.** The Hillstrom file does not record tiredness, and it does not record someone turning notes off after a later message. We write those chances with formulas, the same way the 1.5% cap is a number we chose. Take a buy chance of 0.63. Draw one number between 0 and 1. A draw below 0.63 stores a buy of 1. A draw of 0.63 or higher stores a buy of 0. A second draw does the same for “stop.” The training row stores the 1 or the 0. The trees learn from those answers. We could have handed them 0.63 directly, and a leaf could have averaged the chances of the people who landed there. A real app would have stored the yes or the no, so that is the problem the trees are asked to solve.
4. **On the Shadow price chart, the charge follows the trees and the line is a fresh coin.** Under Results, on Shadow price, there are four charts: opt-out caps of 1.5% and 5%, channel caps of 15% and 30%. Each morning the opt-out charge moves until the trees’ predicted average extra stops sit on the red cap. The cyan and orange lines are a new yes-or-no draw for that same morning, so a line can sit a little off the red cap. The draw is what the chart records. The next morning starts from yesterday’s charge and moves it again because tiredness has changed, and on a weekend the predicted extra stops are multiplied by 1.8 before the charge moves. The Pareto dots are the average of those same daily draws over 21 mornings.
5. **We plan one morning at a time.** The rule looks at today’s tiredness and today’s charges, and scores the people in front of it. It keeps today’s choice separate from the next thirty days. Tomorrow is different because of today, for two reasons. Tiredness has changed, and tomorrow’s tuning starts from the charges today finished with.
6. **The channel charge is a price on a share.** At most 30% of people may get a push this morning, for example. The charge \(\mu_a\) is the price of using that channel. Each person still gets the message with the best score after the prices. The project brief compares this with water-filling: keep handing the better message out until that channel’s share is used up. Here the price \(\mu_a\) is what holds the channel at that share.

The project brief names targets such as how much extra buy is kept, how far opt-outs fall, how much wasted sending is cut, and how often the small copy agrees. This note leaves those as targets. It does not report them as measured results.

---

## 3. Notation and interpretation

This table is a glossary. The names come from section 1, with the gain set to a buy, the loss set to a stop, and the state \(s_{i,t}\) set equal to the tiredness tally \(f_{i,t}\). The five names used most often are the extra buy \(\tau_{\text{conv}}\), the extra stop \(\tau_{\text{fatigue}}\), the tiredness \(f_{i,t}\), the stop charge \(\lambda\), and the channel charge \(\mu_a\).

A chance and a price are different kinds of number, so they get different letters. A hat-\(p\), written \(\hat{p}\), is a predicted chance between 0 and 1. A \(\mu_a\) is the price of using channel \(a\). Older notes wrote the chance as \(\hat\mu\), which looks like the price. This note keeps the chance as \(\hat{p}\).

| Symbol | Plain-language reading | In this project |
| --- | --- | --- |
| \(i = 1, \ldots, N\) | One person in this morning’s list | One held-out customer the rule may contact |
| \(x_i\) | The shopping card known at decision time | Hillstrom fields: recency, history, mens, womens, zip code, newbie, channel |
| \(f_{i,t}\) | Tiredness at the start of morning \(t\) | Starts at 0 on day 1. A message adds a jump. Silence lets it fade |
| \(\mathcal{A}\) | The messages we may send, including silence | Control, Email, Push, In-App Audio. Control is the silence \(a_0\) |
| \(Y(a)\) | The buy recorded if we sent \(a\) | 1 or 0, from a coin flip at the formula’s hidden chance. The chance, such as 0.63, is not stored on the row |
| \(C(a)\) | The “stop” that would happen if we sent \(a\) | 1 or 0, from the declared opt-out formula |
| \(\hat{p}^{\text{buy}}_a(x, f)\) | Predicted chance of a buy if we send \(a\) | A separate group of decision trees for each message. Not a charge |
| \(\hat{p}^{\text{out}}_a(x, f)\) | Predicted chance of “stop” if we send \(a\) | Another group of trees for each message. Not a charge |
| \(\tau_{\text{conv}}(x, f, a)\) | Extra buy versus silence | \(\hat{p}^{\text{buy}}_a - \hat{p}^{\text{buy}}_{\text{Control}}\), once the trees exist. The true gap is the same subtraction on the declared chances |
| \(\tau_{\text{fatigue}}(x, f, a)\) | Extra opt-out versus silence | \(\hat{p}^{\text{out}}_a - \hat{p}^{\text{out}}_{\text{Control}}\). This is not the fatigue tally \(f\) |
| \(\hat\tau\) | The extra the rule actually reads | The tree subtraction above. Section 4 also allows a doubly robust correction of the same subtraction |
| \(B_{\text{opt-out}}\) | The opt-out cap for this morning | A fixed allowance, for example \(1.5\%\). It is not the charge |
| \(C_a\) | The share cap for channel \(a\) | For example \(30\%\). Silence is uncapped |
| \(K\) | How many notes one person may receive in a week | Once the trailing count reaches \(K\), that person can only be left in silence. In this project \(K = 3\) |
| \(\lambda_t \ge 0\) | The opt-out charge at morning \(t\) | Raised when predicted extra opt-out is over \(B_{\text{opt-out}}\), lowered when it is under. Never below 0 |
| \(\mu_a \ge 0\) | The channel charge for message \(a\) | Raised when that channel’s share is over \(C_a\). \(\mu_{\text{Control}} = 0\). Not a probability |
| \(\gamma \in (0, 1)\) | How much of yesterday’s tiredness remains | In this project \(0.85\). The rest fades overnight |
| \(\beta_a \ge 0\) | The tiredness a message adds | Silence 0, email 0.40, push 0.78, audio 0.20 |
| \(\epsilon_{i,t}\) | A tiny wobble on the fatigue update | Average 0. It is not the buy coin flip |
| \(p(a \mid x)\) | The chance the training row drew \(a\) | The draw did not look at the card, so this is the same share for every person. It is not a buy chance and not a charge |

Four kinds of person explain why the rule uses the extra rather than the raw buy chance. The names match the simulator.

- **Persuadable.** The note raises the buy chance, and the extra opt-out is small enough that the score can still win. This is the person the note is for.
- **Sure Thing.** Likely to buy even in silence, so the extra buy is about 0. A ranking by the raw buy chance still puts them near the top. The contrast does not.
- **Lost Cause.** Unlikely to buy under every message, so the extra is about 0. They use up a channel slot if a weak raw score selects them.
- **Sleeping Dog.** The extra buy is negative, or the extra opt-out is so large that the score is negative. The note does harm. Leave them in silence.

The segment is assigned from the shopping card before any coin is drawn. On the file being split into segments, write \(H_{1/4}\), \(H_{1/2}\), and \(H_{3/4}\) for the lower quartile, the median, and the upper quartile of past spend, and \(R_{1/2}\) and \(R_{3/4}\) for the median and the upper quartile of months since the last purchase. A long gap is a large recency. The first matching line wins.

1. **Lost Cause.** New customer, and past spend below the median: \(\text{newbie} = 1\) and \(\text{history} < H_{1/2}\).
2. **Sure Thing.** Past spend in the top quarter, and a last purchase no older than the median gap: \(\text{history} \ge H_{3/4}\) and \(\text{recency} \le R_{1/2}\).
3. **Persuadable.** A long gap, and past spend between the lower and upper quartiles: \(\text{recency} \ge R_{3/4}\) and \(H_{1/4} \le \text{history} \le H_{3/4}\).
4. **Sleeping Dog.** Past spend below the median, and a last purchase no older than the median gap: \(\text{history} < H_{1/2}\) and \(\text{recency} \le R_{1/2}\).

Anyone still unassigned is filled in the same order of ideas. Past spend at or above the median becomes a Sure Thing. A gap longer than the median becomes Persuadable. What remains becomes a Sleeping Dog. The segment then picks the row of the lift table. The trees are not given the segment. They see spend, months, and the new-customer flag.

---

## 4. Predicting contrasts

The rule needs a number for “how many extra buys would this note add, compared with silence?” It cannot see the true answer. It sees an estimate built by histogram gradient-boosted trees. That is a sequence of small decision trees. Each new tree looks at the misses left by the trees already fit, and adds a small correction. The short name for that estimate is \(\hat\tau\), read “tau hat.” The hat means “our estimate of.”

The same extra can be predicted in other ways. A support-vector machine and a neural net are fit on the same rows. Those comparisons are under Results, on Predictors. This section walks through the gradient-boosted trees. Those are the predictors on the prepared charts.

Here is the subtraction on one person. The trees say Ann has a 50% chance of buying after a push and a 40% chance if we stay silent. Her extra buy is \(0.50 - 0.40 = 0.10\). Among 100 people like Ann, a push adds about 10 buys. The 50% by itself does not say that. About 40 of those 50 buyers would have bought with no message.

### One group of trees for each message

The fitting plan is called a T-learner. The T stands for two: one set of trees for people who got a message, and another set for people who were left in silence. This project has four messages, so there is one set of trees per message. “T-learner” is that plan of keeping the silence trees, the email trees, the push trees, and the audio trees in separate groups, and then subtracting. Each group is a set of histogram gradient-boosted trees.

The training samples are split by the one message that was tried. The push-buy trees see only push rows. They learn the buy chance under a push. The silence-buy trees learn the buy chance under silence. Opt-out is fit the same way, so there are eight groups: four messages, and two yes-or-no answers. In symbols, the buy chance under message \(a\) is

\[
\hat{p}^{\text{buy}}_a(x, f)
  = \widehat{P}(B = 1 \mid x, f, a).
\]

The extra buy is that chance minus the silence chance. Silence minus silence is 0:

\[
\hat\tau_{\text{conv}}(x, f, a)
  = \hat{p}^{\text{buy}}_a(x, f) - \hat{p}^{\text{buy}}_{\text{Control}}(x, f),
\qquad
\hat\tau_{\text{conv}}(x, f, \text{Control}) = 0.
\]

The opt-out chance has its own trees, and the extra opt-out is the same subtraction:

\[
\hat{p}^{\text{out}}_a(x, f)
  = \widehat{P}(O = 1 \mid x, f, a),
\]
\[
\hat\tau_{\text{fatigue}}(x, f, a)
  = \hat{p}^{\text{out}}_a(x, f) - \hat{p}^{\text{out}}_{\text{Control}}(x, f).
\]

Those two gaps are what section 5 puts into the score. For Ann’s push, with an extra opt-out of 0.04,

\[
\text{score}(\text{push}) = 0.10 - \lambda \times 0.04 - \mu_{\text{push}}.
\]

\(\lambda\) and \(\mu_{\text{push}}\) are the charges. They are not inside \(\hat{p}\). The trees produce the chances. The subtraction produces the extras. The charges only price the extras.

The same subtraction is why a Sure Thing drops down the list. Ben buys with chance 0.70 in silence and 0.72 under a push. His extra buy is 0.02. A rule that ranks by the raw 0.72 sends him a push he did not need. Cara buys with chance 0.20 in silence and 0.05 under a push. Her extra buy is \(-0.15\). The note talks her out of a purchase. Her score is negative before any charge, so she is left in silence.

One comparison rule skips the subtraction. It ranks people by the raw buy chance \(\hat{p}^{\text{buy}}_a(x, f)\). Anyone who would have bought anyway stays near the top of that list. This rule is one more group of the same trees. The message is written on the row as four columns, one each for silence, email, push, and audio, and the trees learn the buy coin from that row. They do not subtract silence. On the 21-day Pareto charts under Results, this rule is the orange dot, Greedy predictive.

This project uses the separate trees because each message has its own pile of samples, and those piles are large. The push-buy fit is written out below. The other seven fits, silence, email, and audio, for a buy and for an opt-out, follow the same steps on their own rows.

### What a training row contains

One training row is a shopping card from the Hillstrom file, plus the columns we append. The picture is the same one as on the Overview.

White columns come from the file. Orange columns are drawn at random, and that draw ignores the card. Red columns are a yes or a no. The formulas below produce a chance, and a coin flip turns that chance into the red cell. The chances stay off the row.

**The white columns are the shopping card.** Recency is months since the last purchase. History is past spend. Mens is 1 when the person has bought from the men's department, and Womens is the same flag for women. Zip is Rural, Suburban, or Urban. Newbie is 1 for a new customer. Channel is how they used to shop: Multichannel, Phone, or Web. In the picture the card is 10 months, $115 of past spend, no men's purchase, a women's purchase, an urban zip, an established customer, and a past web shopper.

**The orange columns are tiredness and the message.** Tiredness is drawn uniformly between 0 and 3. The message is silence, email, push, or audio, each with the same chance. The pictured row has tiredness 1.2 and a push.

**The buy chance uses the card, the segment, the message, and the tiredness.** The baseline is a number from the card. History enters as a natural log:

\[
b(x) = -3.40 + 0.70 \log(1 + \text{history}) - 0.28\,\text{recency} + 0.12\,\text{mens} + 0.10\,\text{womens}.
\]

For the pictured card that baseline is about \(-2.77\). The segment, assigned in the Notation section, picks a row of the lift table. The message picks a column. The columns are silence, email, push, and audio.

| Segment | Silence | Email | Push | Audio |
| --- | --- | --- | --- | --- |
| Sure Thing | 0 | 0.06 | 0.04 | 0.55 |
| Persuadable | 0 | 2.19 | 3.54 | 1.42 |
| Lost Cause | 0 | 0.04 | 0.02 | 0.05 |
| Sleeping Dog | 0 | \(-0.75\) | \(-1.67\) | \(-0.92\) |

Call the chosen cell \(\ell(x, a)\). Tiredness shrinks it, and the buy chance is a sigmoid of the sum:

\[
p^{\text{buy}}(x, f, a) = \sigma\big(b(x) + \ell(x, a)\, e^{-0.65 f}\big),
\qquad
\sigma(z) = \frac{1}{1 + e^{-z}}.
\]

At tiredness 0 the lift is used in full. A persuadable person whose baseline is about \(-2.69\) then has about a 6% chance in silence and about a 70% chance after a push. At the pictured tiredness of 1.2, \(e^{-0.65 \times 1.2}\) is about 0.46, so that lift is cut to under half before it is added. The segment is used only to pick the row. It is not a column on the row, and the trees do not receive it.

**The stop chance uses the tiredness and the message.** The shopping card stays out of this line.

\[
p^{\text{stop}}(f, a) = \sigma\big(-4.05 + 0.90\, f + 0.40\, f^{2} + c_a\big).
\]

The message cost \(c_a\) is 0 for silence, 0.85 for email, 1.75 for a push, and 0.28 for audio. For the pictured push at tiredness 1.2, the line inside the sigmoid is about \(-0.64\), so the stop chance is about 34%. The training rows are written with no weekend shift, so nothing further is added. On a later weekend the same formula adds \(\log 1.8\), about 0.59, before the sigmoid.

**A coin writes the red columns.** Draw one number between 0 and 1. If it falls below the buy chance, Buy is 1. If it falls at or above that chance, Buy is 0. A second draw does the same for Stop. The pictured row stored a buy and no stop. Those coins are the training target.

Take a buy chance of 0.63 as a smaller example of that same draw. A draw below 0.63 stores \(B = 1\). A draw of 0.63 or higher stores \(B = 0\). The chance of a no is \(1 - 0.63 = 0.37\), and the rule is still one comparison with 0.63. The trees receive the sequence of coins, such as 1, 0, 1, 0. They do not receive the sequence of chances, such as 0.63, 0.57, 0.70, 0.10. A leaf of three people whose hidden chances are 0.63, 0.57, and 0.70 might show coins 1, 0, and 1. The leaf averages the coins, \(2/3\). Averaging the hidden chances would give about 0.63 with no luck in it, and the same cuts-and-gaps machinery would run. The chances stay hidden so the trees recover them from yes-or-no records, which is the record a deployed app would have. Each of these rows is one made-up moment. It is one person at one tiredness, not that person followed for 21 days.

The gap later in this section is measured against that coin. For the first trees,

\[
G = \text{average of }(B - B_{\text{est}}),
\]

where \(B\) is 1 or 0. It is not the average of \((0.63 - B_{\text{est}})\).

### How one group of trees learns a chance

The prepared charts use histogram gradient-boosted trees. The loss is squared error: add up (stored coin minus predicted rate) squared, and look for cuts that make that total smaller. A column is chopped into bins first, and a question picks an edge between bins, such as spend greater than 100. The trees read spend in dollars and months since the last purchase in the units already on the card. Those columns are not rescaled. A support-vector machine and a neural net are also fit on these rows, and they rescale the columns before they fit. Their scores are under Results, on Predictors.

**The starting rate is one number for the whole pile.** Before any tree, every person who was sent a push has the same estimated buy rate \(B_{\text{est}}\). It is the buys in the push pile divided by the people in that pile. About 56,000 people are used for training once the settings are chosen, and about a quarter of them were sent a push, so the pile has about 14,000 rows. If half of those had bought, \(B_{\text{est}}\) would be 0.50 for every one of them. This number is saved. A new person at inference starts from it again. It is not the average of the rates the forty trees later produce.

**One question is one straight cut, and it does not move \(B_{\text{est}}\).** A question uses one field. “Spend greater than 100” is a vertical line if spend is drawn across the page. A later question such as “recency greater than 6” is a horizontal line if recency is drawn up the page. The leaves are the rectangles those lines leave behind. A question cannot tilt: it cannot mix spend and recency in one diagonal cut.

Each side of a cut has its own buy rate, the buys on that side divided by the people on that side. Those two rates are used only to score the question. Suppose 100 people in the current group are still predicted at 0.50, and the cut falls like this:

| Side | People | Buys | Buy rate |
| --- | --- | --- | --- |
| Spend greater than 100 | 50 | 35 | 0.70 |
| Spend at most 100 | 50 | 15 | 0.30 |

The score of a cut is the sum of squared misses. A miss is the actual buy, 1 or 0, minus the rate being used as the prediction, and that difference is squared so a miss of \(+0.3\) and a miss of \(-0.3\) do not cancel. Before this cut, predicting 0.50 for all 100 people gives a squared total of 25. After the cut, predicting 0.70 on one side and 0.30 on the other gives a squared total of 21. Several legal cuts can bring the total down. “Recency greater than 6” might leave 23. The question we keep is the one with the smallest total.

A cut that would leave fewer than 50 people on either side is not a candidate. Five flips of a true 0.50 chance can land as five buys or as none, and the tree would treat 1.00 or 0.00 as the chance in that box. The floor of 50 keeps a lucky handful from becoming a leaf. A finished leaf can hold 50 people or 200. The 50 is a minimum chosen on the validation people, not a promise that the rate is exact.

A path asks at most 3 questions. Each new question is chosen in the same way, on the people who reached that point. Through all of those questions, \(B_{\text{est}}\) stays the rate the pile had at the start of this tree. The group rates choose the cuts. They are not written into \(B_{\text{est}}\) until the tree is finished.

**A finished leaf saves one gap, and everyone in it receives 8% of that gap.** Call the saved gap \(G\). It is the average remaining miss of the training people who landed in the leaf: each person’s actual buy minus that person’s \(B_{\text{est}}\) at the start of this tree. The update is once per finished tree,

\[
B_{\text{est}} \leftarrow B_{\text{est}} + 0.08 \times G.
\]

Tree 1 is the easy case, because everyone still shares one rate. A leaf whose buy rate is 0.80, with the pile still at 0.50, saves \(G = 0.80 - 0.50 = 0.30\). The people who land there become

\[
0.50 + 0.08 \times 0.30 = 0.524.
\]

Tree 2 can put two people in one leaf who arrived with different rates. Ann arrives at 0.54 and bought, so her remaining miss is \(1 - 0.54 = 0.46\). Cal arrives at 0.46 and also bought, so his remaining miss is \(1 - 0.46 = 0.54\). The leaf saves the average of those misses, \(G = 0.50\). Both receive the same nudge, \(0.08 \times 0.50 = 0.04\):

- Ann becomes \(0.54 + 0.04 = 0.58\).
- Cal becomes \(0.46 + 0.04 = 0.50\).

Ann is not moved 8% of her own 0.46, and Cal is not moved 8% of his own 0.54. The leaf remembered one number, and everyone who lands there gets 8% of that one number.

The fitted trees shrink that gap a little before the 8% is applied. The misses in the leaf are added up and divided by the number of people in the leaf plus 1, so a leaf of 50 is divided by 51. The examples above leave that extra 1 out, so the arithmetic stays the plain average. On a leaf of 50 people the fitted gap is about 2% smaller than that average.

The cuts and the gaps are what the training learns. The step of 0.08, the 40 trees, the floor of 50 people, and the limit of 3 questions were the settings with the lowest squared error on the 8,000 validation people. The extra 1 in the division is fixed. The fit always keeps all 40 trees. It does not stop early and throw the later trees away.

**The next tree starts from the updated rates.** Tree 2 holds those new rates still while it chooses its own cuts. When it finishes, it adds 8% of its own saved gaps. Tree 3 starts from the rates tree 2 just wrote.

**Inference walks each person alone through the finished trees.** The trees are not grown again, and the gaps are not recomputed. A new person starts from the saved pile average, the same 0.50 the training rows started from, not from the average of the finished training rates. If that person reaches leaf \(M\) of tree \(N\), whose saved gap is \(G_{NM}\),

\[
B_{\text{est}} \leftarrow B_{\text{est}} + 0.08 \times G_{NM}.
\]

The rate on the right is whatever they have after trees 1 through \(N-1\). Another person scored on the same morning does not change this walk and does not change \(G_{NM}\). Two people who take the same path through all 40 trees finish with the same rate. The finished rate is clipped to lie between 0 and 1. The opt-out trees do this same walk on the opt-out coins. Section 5 then subtracts the silence chance from the push chance, and that difference is the extra the charges price.

### A correction that also uses the stored coin

The push trees learn the push chance only from people who were sent a push, and only from the yes or no on those rows. A lucky run of coins can pull that chance off the true chance.

A second estimate starts from the tree’s chance and then adds a correction from the coin of the people who really received that message. Another group of the same trees learns the corrected number. This second fit is a check. The Shadow price charts and the Pareto charts use the plain subtraction from the first trees.

People sometimes call the second estimate doubly robust. The extra can land in the right place in two ways. The trees’ chance can be right. Or the record of how often each message was sent can be right. One of those is enough.

That record is one share per message. The message was picked by a coin that ignored the shopping card, so silence, email, push, and audio each have about the same share for every card, one person in four. That share is a sending share. It is not a buy chance.

For each person and each message, build a corrected buy number. The mark \(\mathbb{I}(A_i = a)\) is 1 when person \(i\) was sent message \(a\), and 0 otherwise:

\[
\phi_{i,a}
  = \hat{p}^{\text{buy}}_a(x_i, f_i)
  + \frac{\mathbb{I}(A_i = a)}{p(a \mid x_i)}
    \big(B_i - \hat{p}^{\text{buy}}_a(x_i, f_i)\big).
\]

Read it in two pieces. The first piece is the tree’s buy chance. The second piece is the correction. If this person was not sent message \(a\), the mark is 0 and nothing is added. If they were sent \(a\), we add the miss, the stored buy minus the predicted chance, divided by the share who were sent \(a\). One person in four means the miss is multiplied by 4. A rarer message would be counted even more, because each person who received it stands for more people. A second set of trees then learns this corrected number from the card and the tiredness. Call that corrected chance \(\hat{p}^{\text{buy,DR}}_a\). DR is short for doubly robust, the name of this second estimate. The extra is again that chance minus the silence chance:

\[
\hat\tau_{\text{conv}}(x, f, a)
  = \hat{p}^{\text{buy,DR}}_a(x, f) - \hat{p}^{\text{buy,DR}}_{\text{Control}}(x, f).
\]

The same corrected number can be built for a stop, with the stop coin \(O_i\) in place of the buy coin \(B_i\), and the stop chance in place of the buy chance. The chance inside the correction is fit on the other half of the people. The person being corrected is left out of that half. That split is sometimes called cross-fitting. It keeps one person’s own coin from training the chance that then corrects that same coin.

The 21-day charts use the plain subtraction from the first trees, \(\hat{p}^{\text{buy}}_a - \hat{p}^{\text{buy}}_{\text{Control}}\).

The shopping cards do not record a fatigue tally. The simulator attaches a fatigue and an opt-out before these trees are trained. The 21-day test then asks the finished trees again at each morning’s new fatigue.

---

## 5. Constrained policy and shadow prices

The extras from section 4 now meet the house rules. The display below is section 1 with the names filled in. The first line is what we want more of: total extra buys. The next lines are the limits. One cap limits average extra stops. One cap per channel limits the share of people who receive that channel. A third limit is imposed on one person:

\[
\begin{aligned}
\max_{\pi} \quad
& \sum_{i=1}^{N} \sum_{a \in \mathcal{A}}
  \pi(a \mid x_i)\, \hat\tau_{\text{conv}}(x_i, f_i, a) \\
\text{s.t.} \quad
& \frac{1}{N} \sum_{i=1}^{N} \sum_{a \in \mathcal{A}}
  \pi(a \mid x_i)\, \hat\tau_{\text{fatigue}}(x_i, f_i, a)
  \le B_{\text{opt-out}}, \\
& \frac{1}{N} \sum_{i=1}^{N} \pi(a \mid x_i) \le C_a,
  && a \in \mathcal{A} \setminus \{\text{Control}\}, \\
& \pi(\,\cdot \mid x_i) \in \Delta(\mathcal{A}_i).
\end{aligned}
\]

### Two priced caps, and one imposed limit

\(B_{\text{opt-out}}\) is this morning’s allowance for average extra stops, for example 1.5%. \(C_a\) is the share of people who may receive channel \(a\) this morning, for example 30%. Silence is uncapped. Both numbers stay fixed from day to day. They do not appear inside a person’s score. The numbers that appear inside the score are the charges \(\lambda\) and \(\mu_a\). A person with a large extra buy can still pay the charge and take a slot.

The third limit is the weekly count \(K\). In this project \(K = 3\). Count the notes that person has already received in the last seven days. While the count is under 3, the menu is silence, email, push, and audio. Once the count reaches 3, email, push, and audio come off the menu. Silence is the only message left, until an older note falls out of those seven days. On the shadow-price charts this is why some mornings fall near zero: many people are simply not allowed a note that day.

This weekly limit is a constraint that is simply imposed. It is not modeled with a multiplier. The fourth note is that person’s own history, and a large extra buy cannot buy it. Reaching 3 notes does not move \(\lambda\) or \(\mu_a\). The program writes the limit as the last line, \(\pi(\cdot \mid x_i) \in \Delta(\mathcal{A}_i)\). That symbol means a choice of one message from the messages still on that person’s menu. Every extra \(\hat\tau\) is evaluated at today’s card and today’s fatigue.

For a fixed pair of charges, each person is given the message with the largest score. The name for that pick is an argmax, “the action at which the score is largest”:

\[
a_i^*
  = \arg\max_{a \in \mathcal{A}_i}
    \Big[
      \hat\tau_{\text{conv}}(x_i, f_i, a)
      - \lambda\, \hat\tau_{\text{fatigue}}(x_i, f_i, a)
      - \mu_a
    \Big].
\]

Silence scores 0, because both extras are 0 and the silence charge is 0. A message is sent only when its extra buy still wins after the opt-out charge and the channel charge. If every active message scores below 0, the person is left in silence.

The step sizes below are chosen so the \(\lambda\) inside the score is the same \(\lambda\) the update produces. Each person still receives one whole message. After a price pass, a channel or the opt-out total can sit a little outside its cap, because a person cannot be given half a push and half silence. The sorts further down then move whole people.

### Moving the charges

The charges are the control knob. After a scoring pass, let \(\widehat{\text{opt-out}}(\pi)\) be the average extra opt-out of the messages just chosen, as predicted by the trees. Let \(\hat{s}_a(\pi)\) be the share of people assigned to channel \(a\). The share is a fraction of the cohort, such as 0.30. It is not a buy chance, so it is written \(\hat{s}\) rather than \(\hat{p}\).

Compare those two measurements with the caps. If the measurement is over the cap, raise the matching charge. If it is under, lower the charge, but never below 0. The brackets \([z]_{+}\) mean \(\max(z, 0)\):

\[
\begin{aligned}
\lambda
  &\leftarrow
  \Big[
    \lambda
    + \eta_{\lambda}
      \big(\widehat{\text{opt-out}}(\pi) - B_{\text{opt-out}}\big)
  \Big]_{+}, \\[4pt]
\mu_a
  &\leftarrow
  \Big[
    \mu_a
    + \eta_{\mu}
      \big(\hat{s}_a(\pi) - C_a\big)
  \Big]_{+},
  \qquad a \neq \text{Control}.
\end{aligned}
\]

\(\eta_{\lambda}\) and \(\eta_{\mu}\) are step sizes, how far a charge jumps in one pass. In this project they are 2 and 0.12. A step of 18 on \(\lambda\) leaps past the price that meets a loose cap and lands on a much quieter assignment. A step of 2 walks up to that price.

Eighty such passes run inside one morning. The 21-day run starts each morning from the charges left by the previous morning. Each pass lets people switch themselves: a higher \(\lambda\) makes a costly push lose to silence or to a quieter note. The message that is sent is the one left by the last pass, after the two sorts below. The coin-flip buy and opt-out are rolled only then, from the same hidden formulas that wrote the training coins. These are fresh 1s and 0s for the chart. They are not added to the training rows, the trees are not shown the hidden chances, and the flips do not move \(\lambda\) or \(\mu_a\).

### Hard repair

Two sorts finish the morning if a cap is still broken. Both use the scores already computed.

**Channel allowance.** For an active channel \(a\), the integer allowance is \(n_a = \lfloor C_a N \rfloor\), the cap times the number of people, rounded down. Silence is uncapped. Suppose more than \(n_a\) people were given \(a\). For each of them, let \(b_i\) be the next-best message, and let

\[
\mathrm{gap}_i = \mathrm{score}(i, a) - \mathrm{score}(i, b_i).
\]

A small gap means this person barely preferred \(a\). Sort those people by the gap from smallest to largest, and move the smallest gaps onto their next-best message until only \(n_a\) remain. If that next-best channel is itself over its allowance, the same repair visits it. The loop repeats at most once per channel.

**Opt-out allowance.** Let \(\mathrm{cost}_i = \hat\tau_{\text{fatigue}}(x_i, f_i, a_i)\) be the extra opt-out of the message person \(i\) is actually holding, and let

\[
\mathrm{excess} = \sum_{i=1}^{N} \mathrm{cost}_i - N B_{\text{opt-out}}.
\]

If the excess is 0 or negative, the morning already fits and nobody is changed. Otherwise look only at people who are not on silence and whose extra opt-out is positive. Sort them by their chosen score, lowest first. Walk from the bottom and set those people to silence until the extra opt-out that was removed covers the excess. A message that does not raise predicted opt-out is not taken away by this sort.

In plain words: the channel sort takes a buzz, or an email, away from the people least attached to it, and offers them their second choice. The opt-out sort takes the whole note away from the people for whom it was least worth sending, and leaves them in silence.

### From one morning to the next

The same people return the next morning. The shopping card is unchanged. Fatigue has moved. The charges are kept. Tuesday’s twenty passes start from the \(\lambda\) and the \(\mu_a\) that Monday finished with. The caps \(B_{\text{opt-out}}\) and \(C_a\) are the same numbers as Monday. The extras are not, because people are more tired, so the old charges may no longer land on the same caps. Tuesday’s passes move the charges again.

Saturday and Sunday are more expensive. On those mornings every message’s predicted extra opt-out is multiplied by the traffic shift, \(1.8\) in the demo, before the score and the charge updates. The same push now counts as a larger step toward the opt-out cap. If the morning would sit above \(B_{\text{opt-out}}\), the passes raise \(\lambda\) until the predicted extra opt-out is back on the cap. Sunday starts from Saturday’s higher charge.

A second rule, the static weight, is the comparison on the dashboard. On the first morning it copies the \(\lambda\) the closed loop just settled on, and it keeps that one number, called \(\alpha\), for all twenty-one days. Its later score is

\[
\mathrm{score}(a) = \hat\tau_{\text{conv}}(a) - \alpha\, \hat\tau_{\text{fatigue}}(a).
\]

It keeps using Monday’s charge on Saturday, and it scores the extra opt-out before the weekend multiplier. Giving this rule the same updating charges would make it the closed-loop rule. It is frozen so the chart can show a charge that is refit each morning against a charge that is not.

Each morning only settles the people in front of it. Looking past today happens through the fatigue tally: today’s message raises tomorrow’s fatigue, and tomorrow’s extras are computed at that new fatigue.

---

## 6. Fatigue as the horizon

Tiredness is the only number carried from this morning into the next one. The buy and the stop are results of the morning. They are not carried forward. Each person has one tiredness number. The update typed for this project is

\[
f_{i, t+1}
  = 0.85\, f_{i, t}
  + \beta_{a_{i,t}}
  + \epsilon_{i, t}.
\]

The old tally is multiplied by \(0.85\) no matter what we send, so 15% of it fades overnight. \(\beta_a\) is the jump added by the message: 0 for silence, 0.40 for email, 0.78 for a push, and 0.20 for audio. The indicator form in the general write-up, \(\beta_a\) times “the message was not silence,” is the same list, because the silence jump is already 0. \(\epsilon\) is a tiny wobble, a bell-curve draw with standard deviation 0.02. That wobble is not the buy coin flip.

A push that ends in a buy adds the same 0.78 as a push that does not. The coin-flip yes or no is not an input to this line.

The extras depend on fatigue in two directions, both written into the simulator:

- The extra buy of an active message shrinks as fatigue grows. A tired person yields less from another note.
- The extra opt-out grows as fatigue grows, and it grows faster than a straight line. The chance of “stop” rises slowly at first and then steeply.

The rule sees today’s fatigue when it scores. A push can win Monday’s score and still be a poor note, because it raises Tuesday’s fatigue, which shrinks Tuesday’s extra buy and raises Tuesday’s extra opt-out. Leaving the person in silence lets fatigue fade.

These numbers \(\gamma = 0.85\) and the jumps \(\beta_a\) are settings we chose. They are not fit from the shopping cards.

---

## 7. Distillation

The morning’s choice is heavy. It asks eight groups of trees, scores every message, and then moves the charges. Call that full choice the teacher. It is the decision we trust.

The student is a small boosted tree trained to copy the teacher. It reads only the shopping card and the tiredness, and it is trained to name the message the teacher assigned at this morning’s charges. Asking the student is the speed check: how fast can one small tree repeat the teacher’s choice, and how often does it name the same message.

Agreement and speed are checks on the copy. They do not change who receives which note. The executive spec’s agreement and speed figures are the targets for those checks. This note does not report them as measured results.

---

## 8. The same problem as content decisioning

The same morning’s choice can decide what to do with a song or a post, once some other model has already scored it. The “person” is then a piece of content, or a track meeting a listener. The card \(x\) holds whatever that earlier model produced: tags, a quality score, the creator’s history, and which screen this is. The menu is how the content is treated, for example

\[
\mathcal{A}
  = \{\text{distribute},\, \text{downrank},\, \text{hold for review},\, \text{no-op}\},
\]

“Do nothing” is the reference, the way silence is the reference for a message. The extra value is the change in listener value compared with doing nothing. The extra cost is the change in risk, harm, or review load compared with doing nothing. The budgets are operating limits: a ceiling on harm, a review queue that can hold only so many items, a limit on how often an automatic action fires. The charge \(\lambda\) is then the price of predicted risk, and \(\mu_a\) is the price of a review slot. The rule is the same pick as in section 5: take the action with the best extra value after those prices.

Classifiers and rankers are one way to produce those extras. They play the role the trees play in section 4. This note starts once the extras exist and a choice has to be made under a budget.

This repository fills the action in as a message, and the cost in as notification fatigue. It does not train a content encoder, a moderation classifier, or a catalog-enforcement model. The mapping above is why section 1 left the names open.
