/* Dataset section */
const data_array = [
    {
        "query": "The rise of large language models (LLMs) has had a significant impact on various domains, including natural language processing and artificial intelligence. While LLMs such as ChatGPT have been extensively researched for tasks such as code generation and text synthesis, their application in detecting malicious web content, particularly phishing sites, has been largely unexplored. To combat the rising tide of automated cyber attacks facilitated by LLMs, it is imperative to automate the detection of malicious web content, which requires approaches that leverage the power of LLMs to analyze and classify phishing sites. In this paper, we propose a novel method that utilizes ChatGPT to detect phishing sites. Our approach involves leveraging a web crawler to gather information from websites and generate prompts based on this collected data. This approach enables us to detect various phishing sites without the need for fine-tuning machine learning models and identify social engineering techniques from the context of entire websites and URLs. To evaluate the performance of our proposed method, we conducted experiments using a dataset. The experimental results using GPT-4 demonstrated promising performance, with a precision of 98.3% and a recall of 98.4%. Comparative analysis between GPT-3.5 and GPT-4 revealed an enhancement in the latter's capability to reduce false negatives. These findings not only highlight the potential of LLMs in efficiently identifying phishing sites but also have significant implications for enhancing cybersecurity measures and protecting users from the dangers of online fraudulent activities.",
        "results": [
            {
                "論文のタイトル": "Detecting Phishing Sites Using ChatGPT",
                "概要": "大規模言語モデルChatGPTを使用してフィッシングサイトを検出する新しい方法を提案。ウェブクローラーを用いて情報を収集し、機械学習モデルを微調整することなくフィッシングサイトを検出する。GPT-4を使用した実験結果は精度98.3％、再現率98.4％を示す。",
                "長所": "ChatGPTの高い検出性能と、微調整なしでフィッシングサイトを検出できる点。",
                "短所": "GPT-4以前のモデルと比較した場合、誤検出（偽陰性）が増加する可能性がある。",
            }, 
            {
                "論文のタイトル": "Know Your Phish: Novel Techniques for Detecting Phishing Sites and their Targets",
                "概要": "ウェブページの特性を利用した新しいフィッシング検出システムを提案。ターゲットの識別コンポーネントを開発し、偽陽性を最小化する。",
                "長所": "クライアントサイドで実装可能で、言語に依存しない。速度が速く、優れた分類性能を持つ。",
                "短所": "大量の学習データが必要で、新たなターゲットに対するフィッシング攻撃の検出には不適当かもしれない。",
            },
            {
                "論文のタイトル": "URLTran: Improving Phishing URL Detection Using Transformers",
                "概要": "トランスフォーマーモデルを使用してフィッシングURL検出タスクのパフォーマンスを向上させるURLTranを提案。対抗策としての悪意のある攻撃の改善にも取り組んでいる。",
                "長所": "従来の深層学習ベースの方法と比較して、非常に低い偽陽性率(FPR)でフィッシングURL検出のパフォーマンスを大幅に向上させる。",
                "短所": "一部の固定攻撃に対しては、偽陽性率が上昇する可能性がある。",
            },
            {
                "論文のタイトル": "Look Before You Leap: Detecting Phishing Web Pages by Exploiting Raw URL And HTML Characteristics",
                "概要": "WebPhishというフィッシング攻撃の検出を目指した深層学習技術を提案。URLとHTMLの生の特性を利用して特徴を抽出し、それらをCNNモデルの入力として利用。",
                "長所": "実世界のデータセットでの実験結果は98％の精度を達成。完全に言語に依存しないクライアントサイドの戦略であり、ウェブページのテキスト言語に関係なく軽量なフィッシング検出が可能。",
                "短所": "新しいデータへの学習の外挿が難しい。従来の方法と比較して特徴選択の自動化が必要。",
            },
            {
                "論文のタイトル": "Detecting Phishing sites Without Visiting them",
                "概要": "エンドユーザーがサイトを訪れることなくサイトの真正性を検出できる方法を提案。訓練とモデル構築のために6つの異なる分類器を使用。ブラウザ拡張機能として開発され、リンクの上にカーソルを置くとポップアップが表示され、ウェブサイトの性質を表示。",
                "長所": "訪問せずにフィッシングサイトを検出できる。複数の機械学習アルゴリズムを用いて分類性能を向上させる。",
                "短所": "システムの設定により、フィッシングサイトの検出が遅延する可能性がある。また、95％の最高精度は他の手法と比較して若干低い。",
            },
            /*
            {
                "論文のタイトル": "",
                "概要": "",
                "長所": "",
                "短所": "",
            }
            */
        ],
    },
    {
        "query": "test1",
        "results": [],
    },

    {
        "query": "test2",
        "results": [
            {
                "論文のタイトル": "test2",
                "概要": "概要",
                "長所": "長所",
                "短所": "短所",
            }
        ],
    },
];

const query_default = data_array[0]["query"]
//queryに対して論文情報をまとめた結果を返す. 
const get_results = async (query=query_default) => {
    const sleep = (sec) => { return new Promise(resolve => setTimeout(resolve, sec*3000)) }
    await sleep(1);

    //本来はサーバーにリクエストを投げる. デモとしてここではローカルに完結する. 
    for (let data of data_array) {
        if (data["query"]===query) {
            return data["results"]
        }
    };
    throw new Error("data not found!");
    return {};
}


const generateTable = async (query="") => {
    try{
        // datasets
        let headers = ["論文のタイトル","概要","長所","短所"];
        let values = await get_results(query);
        //console.debug(values);


        let userInput = document.getElementById('search-input').value;

        // table
        let table = document.createElement('table'); 
        table.border = 1;
        table.style.borderCollapse = "collapse";

        //thead
        let thead = document.createElement('thead');
        let headerRow = document.createElement('tr');
        for (let col_i=0; col_i<headers.length; col_i++) {
            let headerCell = document.createElement('th');
            let header = headers[col_i];
            headerCell.textContent = header;
            headerRow.appendChild(headerCell);
        }
        thead.appendChild(headerRow);
        table.appendChild(thead);
            
        //tbody 
        let tbody = document.createElement('tbody');
        for (let row_i = 0; row_i < values.length; row_i++) {
            let value = values[row_i];
            let row = document.createElement('tr');
            for (let col_i=0; col_i<headers.length; col_i++) {
                let cell = document.createElement('td');
                let header = headers[col_i];
                cell.textContent = value[header];
                row.appendChild(cell);
            }
            tbody.appendChild(row);
        }
        table.appendChild(tbody);

        return table;
    }catch(e){
        console.error(e);
        let div = document.createElement("div");
        div.innerHTML = `
            <font color="red">
                <p> Free search is not available now. </p>
                <p> Only following queries can be searched. </p>
                <ul>
                ${
                    data_array.map((data)=>{
                        return `<li>${data["query"]}</li>`;
                    })
                }
                </ul>
            </font>
        `;
        return div;
    }
    
}

