import type { NextApiRequest, NextApiResponse } from 'next';
import WebSocket from 'ws';

function connectWebSocket(
  question: string,
  history: any,
  res: NextApiResponse,
) {
  const ws = new WebSocket(process.env.WS_CHAT_API_URL!);
  let readyToSendToken = !history || history.length === 0;
  let promptCount = 0;
  const waitForPromptCount = process.env.WAIT_FOR_PROMPT_COUNT
    ? parseInt(process.env.WAIT_FOR_PROMPT_COUNT)
    : 2;

  const sendData = (data: string) => {
    res.write(`data: ${data}\n\n`);
  };

  ws.onopen = function () {
    console.log('socket.onopen');
    const msg = { question, history };
    ws.send(JSON.stringify(msg));
  };

  ws.onmessage = function (e: any) {
    // console.log('Message:', e.data);
    let parsedData = JSON.parse(e.data);
    const result = parsedData.result;
    if (
      !result ||
      result.length == 0 ||
      (result.length > 20 && result[0] !== '{')
    ) {
      if (result && result.length) {
        console.log('onmessage:', result);
      }
      if (result && result.startsWith('Prompt after formatting:')) {
        if (!readyToSendToken) {
          promptCount++;
          if (promptCount === waitForPromptCount) {
            readyToSendToken = true;
          }
        }
      }
      return;
    }

    if (result.length > 2 && result[0] == '{') {
      console.log('\n\n', result);
      sendData(result);
    } else {
      process.stdout.write(result);
      if (readyToSendToken) {
        sendData(JSON.stringify({ token: result }));
      }
    }
  };

  ws.onclose = function (e: any) {
    console.log('Socket is closed.', e.reason);
    res.end();
  };

  ws.onerror = function (err: any) {
    console.error('Socket encountered error: ', err);
    ws.close();
  };
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  if (req.method === "OPTIONS") {
    return res.send("ok");
  }

  console.log('req.body: ', req.body);
  const { question, history } = req.body;

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });

  connectWebSocket(sanitizedQuestion, history, res);
}
