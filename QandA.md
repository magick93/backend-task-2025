# Backend Take-Home Task

## Questions

| General \- checking assumptions |  |  |  |
| :---- | :---- | :---- | :---- |
| ![No type][image1] Question | ![No type][image1] Answer | ![People][image2] Owner | ![Drop-downs][image3] Status |
| Can we clarify the objective / value of this project (excluding that of accessing my skillings and thinking). Am I correct in thinking that it is about the discovery of sub themes within provided themes? Eg, within the `accounts` theme we might discover \`login error\` issues. | Yes, it is to provide a reactive / fully generated level of insights beyond what is provided by the structured/consistent themes | Person | Not started |
|  |  | Person | Not started |
| Based on the following:\> Sentences come from comments, ids are from comments, therefore there can be multiple sentences with the same id. This is expected and correct data.  I’m assuming that the data has already gone through a data prep phase. Is it reasonable to make the following assumptions too: Input data is already stored (somewhere) and this service does not need persist the input data Data is considered clean? Eg, offensive, PII, etc is removed | No data should be persisted.  Yes, although some LLMs make their own decisions about this. So if you use an LLM it may need to be considered | Person | Not started |
| Again, in terms of assumed usage, should I assume this is intended for production, and not experimentation? The key difference being that the experimentation approach is interested in finding the best models, recording various model output data for comparison, whereas a production system focuses on performance, reliability, security etc.  | This is designed for production. Although production always needs an eval framework (but you only have 4 hours so you might not want to include that). | Person | Not started |

| Input & Output Details |  |  |  |
| :---- | :---- | :---- | :---- |
| ![No type][image1] Question | ![No type][image1] Answer | ![People][image2] Owner | ![Drop-downs][image3] Status |
| For sentiment analysis: Should this be determined at cluster level (aggregate of all sentences) or based on key insights? Are we scoring sentiment for each sentence individually first? | You can decide this, but we do it on a cluster level | Person | Not started |
| Are the IDs expected to be in uuid format?  | yes | Person | Not started |
| Should non-uuid ids be rejected? | Sure, although they can be seen as just strings for the purpose of this. | Person | Not started |
| It seems that the data, while probably synthetic, is based on comments from the google and apple app stores. These comments also have dates, star ratings and possible developer response. Is it of value to at least model these too? | Not for this exercise… I’ve tried to constrain the input/output | Person | Not started |
| Should we log and reject non-english input? | Depends on your approach :-)  | Person | Not started |
| Themes \- the sample data is good, but is it possible to get the complete list of themes? | I’ve tried to constrain the input/output | Person | Not started |
| Are there any sub-themes that are of low or no value and therefore should be ignored? | I’m interested in seeing your thinking about this kind of thing. Even flagging things like this for discussion would be good. | Person | Not started |

| AI/Model Requirements |  |  |  |
| :---- | :---- | :---- | :---- |
| ![No type][image1] Question | ![No type][image1] Answer | ![People][image2] Owner | ![Drop-downs][image3] Status |
| Is there a preference to use specific models (like Hugging Face, AWS Comprehend, OpenAI) or choose our own? Or is the preference for loose coupling of models / model agnostic? | I wouldn’t focus too hard on model selection. We do have constraints internally but I’m not projecting those onto your work | Person | Not started |
| Is there a preference for hosted vs. self-hosted models considering cost, latency, and data privacy? | Cost, latency, and data privacy are always a concern\! I can say that the real version of this problem had a 10s budget and had to cost less than $0.005. And use models available in australia without data leaving the country. But you don’t have to necessarily follow these | Person | Not started |
| What's the accuracy/quality expectation? Are we optimizing for speed (sub-10s) or quality of clustering? | These are the tradeoffs and looking forward to discussing\! But the real version of this feature is designed to be web-real time (i.e. someone is looking at a spinner) | Person | Not started |
| I’m assuming it would be of value to be able to run different models and capture the derived output.  Is it work including, or at least considering for future use, tools like MLFlow or LangFuse? | I will leave this up to your approach / what you want to focus on. | Person | Not started |
|  |  | Person | Not started |
|  |  | Person | Not started |

| Infrastructure, operations and compliance |  |  |  |
| :---- | :---- | :---- | :---- |
| ![No type][image1] Question | ![No type][image1] Answer | ![People][image2] Owner | ![Drop-downs][image3] Status |
| Are there compliance requirements (GDPR, HIPAA) affecting data storage/transit? | Yes, but probably not in scope to go too deep | Person | Not started |
| Is it of value to clean any PII data from comments? I’m assuming this would be low value as it's likely already done at the appstore level.  | This can be assumed to be done already | Person | In progress |
| Are there concerns or policies regarding data being sent to 3rd parties? Is data sovereignty a concern? | Yes and yes in the real world. But not in scope to go too deep | Person | Blocked |
| Is there a preference of IaC and CI/CD approaches?Terraform in github workflow, or value both DX and CI/CD independently? | I’m interested in seeing your approach\! | Person | Approved |

| Actors |  |  |  |
| :---- | :---- | :---- | :---- |
| ![No type][image1] Topic | ![No type][image1] Decision | ![People][image2] Owner | ![Drop-downs][image3] Status |
| Can you tell me about who or what will be providing the inputs and consuming the outputs | Assume it will be an api layer that has prepared data and passed it to this service | Person | Not started |
| Will it be another service that provides the input, or a human? Will it be run on schedule, or on demand? | On demand | Person | Not started |
| What kind of auth does the actor currently use, eg jwt, saml, etc | To simplify, this is a service behind an already authenticated layer | Person | Not started |
| Is there a need for resource restriction based on role, eg, certain roles can run certain models. I’m assuming not. | To simplify, this is a service behind an already authenticated layer | [Anton](mailto:kurrent93@gmail.com) | Not started |
| Does the consumer and producer expect the response and request to be of a specific shape / type, eg \`hateos\` | There will definitely need to be consistent output | Person | Not started |





