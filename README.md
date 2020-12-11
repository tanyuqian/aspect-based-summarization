# Summarizing Text on Any Aspects

This repo contains preliminary code of the following paper:

[Summarizing Text on Any Aspects: A Knowledge-Informed Weakly-Supervised Approach](https://arxiv.org/abs/2010.06792) \
Bowen Tan, Lianhui Qin, Eric P. Xing, Zhiting Hu \
EMNLP 2020 

## Getting Started
Given a document and a target aspect (e.g.,
a topic of interest), aspect-based abstractive
summarization attempts to generate a summary with respect to the aspect. Previous studies usually assume a small pre-defined set of
aspects and fall short of summarizing on other
diverse topics. In this work, we study summarizing on arbitrary aspects relevant to the document, which significantly expands the application of the task in practice. 
Due to the lack
of supervision data, we develop a new weak
supervision construction method and an aspect
modeling scheme, both of which integrate rich
external knowledge sources such as ConceptNet and Wikipedia. Experiments show our approach achieves performance boosts on summarizing both real and synthetic documents
given pre-defined or arbitrary aspects.

## Weak Supervision Construction
We construct weak supervisions from CNN/DailyMail dataset (Hermann et al., 2015) 
which contains ~300K ```(document, summary)``` pairs.

### Download the Constructed Dataset

The constructed dataset is available [here](https://drive.google.com/file/d/17ZeJsxyottRyvfzguedoET7OFkWSgJJK/view?usp=sharing). 

It contains ~4M ```(document, aspect, summary)``` triples stored in JSON format including these keys:
* ```document```: the document from CNN/DM dataset.
* ```global summary```: the generic summary from CNN/DM dataset.
* ```aspect```: an extracted aspect.
* ```summary```: constructed summary with respect the aspect.
* ```reasoning```: one or multiple knowledge terms from ConceptNet knowledge graph, indicating reasons to construct the summary given the aspect. Each term is in the format of ```ConceptNet: [[head entity]] relation [[tail entity]]```.
* ```important words```: extracted (<= 20) words from the document which are most related to the aspect. We use the Wikipedia page of the
aspect for filtering the words. (If Wikipedia does not have a page of the aspect, it would be an empty list here.)

An example is as below. 
```
[
    {
        "document": "LONDON, England (CNN) -- French Foreign Minister Bernard Kouchner's declaration that France had to prepare for the possibility of war against Iran over its nuclear program was not conventional diplomatic behavior. But then Kouchner was never expected to be a soft-soaper on the diplomatic scene. French foreign minister Bernard Kouchner has a reputation for challenging convention and authority. A surprise appointment from the Socialist ranks to Nicolas Sarkozy's conservative government, the founder of Medicins Sans Frontiers has always challenged convention and authority. The former UN Secretary General Boutros Boutros-Ghali once called Kouchner 'an unguided missile' and the man himself has been known to declare: \"To change the law you sometimes have to break the law\". He was in his youth one of the leaders of the students revolt in France in May 1968. Kouchner is a humanitarian as well as a patriot, with a strong commitment to human rights. Unusually for a man of the Left, he supported the US-led intervention in Iraq (while criticizing the aftermath). But he did so on the grounds of Saddam Hussein's denial of human rights, not his possible possession of weapons of mass destruction. His and President Sarkozy's concern for human rights lies behind their eagerness to join Gordon Brown's Britain in a new push for action in Darfur. Bernard Kouchner did not come to his position with any of former President Chirac's instinctive distrust of the United States. Washington, which has been critical of some European states for their weakness in confronting Teheran, will have been delighted by his 'get serious' warning to Teheran. But the plain-speaking Kouchner is unlikely to be deterred by fears of upsetting the White House when he has criticisms to make of US policy. How much should be made of his words on Iran remains unclear at this stage. They were scarcely on the same scale as President Chirac's threat when he was still in office to retaliate with nuclear strikes against any state found to be responsible for a large-scale terrorist attack on France. But they are all of a piece with France's new high-profile style under the presidency of Nicolas Sarkozy. Mr Kouchner, for example, became the first French Foreign Minister to visit Iraq since 1988, insisting that there could only be a political solution to the country's problems, not a military one, and offering France's services as a mediator and 'honest broker' between Sunnis, Shiites and Kurds. On Iran he is, in a way, merely echoing the words of his President who declared in a speech last month that a nuclear-armed Iran would be 'unacceptable' and describing the stand-off over its nuclear program as 'undoubtedly the most serious crisis before us today'. Certainly Mr Kouchner is making clear that France no longer takes the view once expressed by President Chirac that a nuclear-armed Iran might be inevitable. In continuing to ratchet up the rhetoric over that threat and to underline the West's resolution on Iran's nuclear enrichment program Mr Kouchner is supplementing his president's warnings. Neither is saying that military intervention against Iran is imminent or inevitable. Neither has yet confirmed that France would be part of any such military action. But both are stressing the risks which are piling up as a result of Teheran's brinkmanship. Perhaps the strongest lesson though from Mr Kouchner's intervention is his underlining that the new administration in France is not a knee-jerk anti-American one -- and that France is in the business of reclaiming a role at the top diplomatic tables. E-mail to a friend.",
        "global summary": "French FM Kouchner has told France to prepare for possibility of war with Iran. Was a surprise appointment to Nicolas Sarkozy's conservative government. Also the first French Foreign Minister to visit Iraq since 1988. Founder of Medicins Sans Frontiers, also  French student leader in May 1968.",
        "aspect summaries": [
            {
                "aspect": "france",
                "summary": "French FM Kouchner has told France to prepare for possibility of war with Iran. Was a surprise appointment to Nicolas Sarkozy's conservative government. Also the first French Foreign Minister to visit Iraq since 1988. Founder of Medicins Sans Frontiers, also  French student leader in May 1968.",
                "reasonings": "ConceptNet: [[french]] HasContext [[france]]; ConceptNet: [[nicolas_sarkozy]] RelatedTo [[france]]",
                "important_words": ["france", "nuclear", "sarkozy", "bernard", "president", "intervention", "diplomatic", "nicolas", "french", "convention", "foreign", "rights", "military", "human", "minister", "scale", "iraq", "authority", "armed", "threat" ]
            },
            {
                "aspect": "iran",
                "summary": "French FM Kouchner has told France to prepare for possibility of war with Iran. Also the first French Foreign Minister to visit Iraq since 1988.",
                "reasonings": "ConceptNet: [[war]] RelatedTo [[iran]]; ConceptNet: [[iraq]] RelatedTo [[iran]]",
                "important_words": ["iran", "france", "nuclear", "president", "diplomatic", "program", "foreign", "rights", "military", "human", "minister", "scale", "iraq", "authority", "frontiers", "armed", "kurds", "revolt", "saddam", "action"]
            },
            ...
        ]
    },
    ...
]
```

### Build you own dataset
We provide our code to construct our dataset in [weak_supervision_construction/](weak_supervision_construction/), including handy APIs of ConceptNet and Wikipedia.
Feel free to play with it:)

## Model

*code coming soon...*
