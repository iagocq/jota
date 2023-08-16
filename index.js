const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

var allowed_chats = [
    /^Saia saia do grupo$/,
    /^DCC da Depress/,
    /^Criazada da facul$/,
];

var last_messages = {};
var allowed_chat_ids = [];

const client = new Client({
    authStrategy: new LocalAuth(),
});

client.on('qr', (qr) => {
    // Generate and scan this code with your phone
    qrcode.generate(qr, { small: true });
});

client.on('ready', async () => {
    console.log('Client is ready!');
    let chats = await client.getChats();

    chats.forEach(chat => {
        if (allowed_chats.some(regex => regex.test(chat.name))) {
            allowed_chat_ids.push(chat.id._serialized);
        }
    });

    console.log(allowed_chat_ids);
});

function isAllowedChat(chat_id) {
    // return allowed_chat_ids.includes(chat_id);
    return true;
}

async function fetchJson(url, data) {
    return await fetch(url, {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        },
    }).then(res => res.json());
}

async function answerPrompt(prompt, context) {
    return await fetchJson('http://localhost:5000/prompt', { prompt: prompt, context: context });
}

async function answerQuery(query) {
    return await fetchJson('http://localhost:5000/query', { query: query });
}

client.on('message', async msg => {
    if (!isAllowedChat(msg.from)) return;
    let last_message = last_messages[msg.from] || '';
    last_messages[msg.from] = msg.body;

    let chat = await msg.getChat();

    if (msg.body.length > 2 && (!chat.isGroup || msg.body.match(/^jota/i))) {
        let reply = msg.hasQuotedMsg ? (await msg.getQuotedMessage()).body : '';

        await chat.sendStateTyping();
        let response = await answerPrompt(msg.body, reply || last_message);

        await chat.clearState();
        await msg.reply(response.answer);
        last_messages[msg.from] = response.answer;

        // await client.sendMessage(msg.from, `Gerado com o seguinte comando SQL:\n\`\`\`${response.sql}\`\`\``);
    } else if (msg.body.match(/^!sql /i)) {
        await chat.sendStateTyping();
        let response = await answerQuery(msg.body.replace(/^!sql /i, ''));
        await chat.clearState();
        await msg.reply(response.results);
    }
});

client.initialize();
