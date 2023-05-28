import { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import Layout from '@/components/layout';
import styles from '@/styles/Home.module.css';
import { Message } from '@/types/chat';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import Image from 'next/image';
import ReactMarkdown from 'react-markdown';
import LoadingDots from '@/components/ui/LoadingDots';
import { Document } from 'langchain/document';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { finished } from 'stream';

const chatApiUrl = process.env.NEXT_PUBLIC_DOCS_CHAT_API_URL || '';
const toUseWebSocket = chatApiUrl.startsWith('ws');

export default function Home() {
  const [query, setQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [ready, setIsReady] = useState<boolean>(false);
  const [sourceDocs] = useState<Document[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [messageState, setMessageState] = useState<{
    messages: Message[];
    pending?: string;
    lastQuestion?: string;
    history: [string, string][];
    pendingSourceDocs?: Document[];
  }>({
    messages: [
      {
        message:
          process.env.NEXT_PUBLIC_HELLO ||
          'Hi, what would you like to experience?',
        type: 'apiMessage',
      },
    ],
    history: [],
    pendingSourceDocs: [],
  });

  const { messages, pending, history, pendingSourceDocs } = messageState;

  const messageListRef = useRef<HTMLDivElement>(null);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  const webSocket = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!loading) {
      textAreaRef.current?.focus();
    }
  }, [loading]);

  async function handleData(data: any) {
    try {
      let parsedData = JSON.parse(data);
      const result = parsedData.result;
      if (result !== undefined) {
        if (result.length == 0 || (result.length > 20 && result[0] !== '{')) {
          return;
        }
        parsedData.token = result;

        try {
          if (result.length > 2 && result[0] == '{') {
            parsedData = JSON.parse(result);
          }
        } catch (error) {
          // ignore
        }
      }

      if (parsedData.token) {
        setMessageState((state) => ({
          ...state,
          pending: (state.pending ?? '') + parsedData.token,
        }));
      }

      let finished = false;
      if (parsedData.sourceDocs) {
        finished = true;
        setMessageState((state) => ({
          ...state,
          pendingSourceDocs: parsedData.sourceDocs,
        }));
      } else if (parsedData.error) {
        finished = true;
        setMessageState((state) => ({
          ...state,
          pending: (state.pending ?? '') + parsedData.error,
        }));
      }

      if (finished) {
        setMessageState((state) => ({
          history: [
            ...state.history,
            [state.lastQuestion!, state.pending ?? ''],
          ],
          messages: [
            ...state.messages,
            {
              type: 'apiMessage',
              message: state.pending ?? '',
              sourceDocs: state.pendingSourceDocs,
            },
          ],
          pending: undefined,
          pendingSourceDocs: undefined,
          lastQuestion: undefined,
        }));
        setLoading(false);
      }
    } catch (error) {
      console.log('handleData error:', error);
    }
  }

  function connectWebSocket() {
    if (webSocket.current) {
      return;
    }
    const ws = new WebSocket(chatApiUrl);
    webSocket.current = ws;

    ws.onopen = function () {
      console.log('socket.onopen');
      setIsReady(true);
    };

    ws.onmessage = function (e) {
      console.log('Message:', e.data);
      handleData(e.data);
    };

    ws.onclose = function (e) {
      webSocket.current = null;
      setIsReady(false);

      console.log(
        'Socket is closed. Reconnect will be attempted in 1 second.',
        e.reason,
      );
      setTimeout(function () {
        connectWebSocket();
      }, 1000);
    };

    ws.onerror = function (err) {
      console.error('Socket encountered error: ', err);
      ws.close();
    };
  }

  useEffect(() => {
    if (toUseWebSocket && !webSocket.current) {
      connectWebSocket();
    }
  });

  //handle form submission
  async function handleSubmit(e: any) {
    e.preventDefault();

    setError(null);

    if (!query) {
      alert('Please input a question');
      return;
    }

    const question = query.trim();

    setMessageState((state) => ({
      ...state,
      messages: [
        ...state.messages,
        {
          type: 'userMessage',
          message: question,
        },
      ],
      pending: undefined,
      lastQuestion: question,
    }));

    setLoading(true);
    setQuery('');
    setMessageState((state) => ({ ...state, pending: '' }));

    const ctrl = new AbortController();

    try {
      if (toUseWebSocket) {
        if (webSocket.current && ready) {
          const msg = { question, history };
          webSocket.current.send(JSON.stringify(msg));
        }
      } else {
        await fetchEventSource(chatApiUrl || '/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question,
            history,
          }),
          signal: ctrl.signal,
          onmessage(event) {
            handleData(event.data);
          },
          onclose() {
            console.log('Connection closed by the server');
            ctrl.abort();
          },
          onerror(err) {
            console.log('There was an error from server', err);
          },
        });
      }
    } catch (error) {
      setLoading(false);
      setError('An error occurred while fetching the data. Please try again.');
      console.log('error', error);
    }
  }

  const onSubmit = useCallback(handleSubmit, [query]);

  //prevent empty submissions
  const handleEnter = useCallback(
    (e: any) => {
      if (e.key === 'Enter' && query) {
        handleSubmit(e);
      } else if (e.key == 'Enter') {
        e.preventDefault();
      }
    },
    [query],
  );

  const chatMessages = useMemo(() => {
    return [
      ...messages,
      ...(pending
        ? [
          {
            type: 'apiMessage',
            message: pending,
            sourceDocs: pendingSourceDocs,
          },
        ]
        : []),
    ];
  }, [messages, pending, pendingSourceDocs]);

  //scroll to bottom of chat
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [chatMessages]);

  return (
    <>
      <Layout>
        <div className="mx-auto flex flex-col gap-4">
          <h1 className="text-2xl font-bold leading-[1.1] tracking-tighter text-center">
            {process.env.NEXT_PUBLIC_TITLE || 'Chat with Mastercard Priceless'}
          </h1>
          <main className={styles.main}>
            <div className={styles.cloud}>
              <div ref={messageListRef} className={styles.messagelist}>
                {chatMessages.map((message, index) => {
                  let icon;
                  let className;
                  if (message.type === 'apiMessage') {
                    icon = (
                      <Image
                        key={index}
                        src="/bot-image.png"
                        alt="AI"
                        width="40"
                        height="40"
                        className={styles.boticon}
                        priority
                      />
                    );
                    className = styles.apimessage;
                  } else {
                    icon = (
                      <Image
                        key={index}
                        src="/usericon.png"
                        alt="Me"
                        width="30"
                        height="30"
                        className={styles.usericon}
                        priority
                      />
                    );
                    // The latest message sent by the user will be animated while waiting for a response
                    className =
                      loading && index === chatMessages.length - 1
                        ? styles.usermessagewaiting
                        : styles.usermessage;
                  }
                  return (
                    <>
                      <div key={`chatMessage-${index}`} className={className}>
                        {icon}
                        <div className={styles.markdownanswer}>
                          <ReactMarkdown linkTarget="_blank">
                            {message.message}
                          </ReactMarkdown>
                        </div>
                      </div>
                      {message.sourceDocs && (
                        <div
                          className="p-5"
                          key={`sourceDocsAccordion-${index}`}
                        >
                          <Accordion
                            type="single"
                            collapsible
                            className="flex-col"
                          >
                            {message.sourceDocs.map((doc, index) => (
                              <div key={`messageSourceDocs-${index}`}>
                                <AccordionItem value={`item-${index}`}>
                                  <AccordionTrigger>
                                    <h3>
                                      {process.env.NEXT_PUBLIC_SOURCE ||
                                        'Source'}{' '}
                                      {index + 1}
                                    </h3>
                                  </AccordionTrigger>
                                  <AccordionContent>
                                    <ReactMarkdown linkTarget="_blank">
                                      {doc.pageContent || doc.page_content}
                                    </ReactMarkdown>
                                    <p className="mt-2">
                                      <b>
                                        {process.env.NEXT_PUBLIC_URL || 'URL'}:
                                      </b>{' '}
                                      <a
                                        target="_blank"
                                        href={doc.metadata.url}
                                      >
                                        {doc.metadata.url}
                                      </a>
                                    </p>
                                    {doc.metadata.page !== undefined && (
                                      <p className="mt-2">
                                        <b>
                                          {process.env.NEXT_PUBLIC_PAGE ||
                                            'Page'}
                                          :
                                        </b>{' '}
                                        {doc.metadata.page + 1}
                                      </p>
                                    )}
                                  </AccordionContent>
                                </AccordionItem>
                              </div>
                            ))}
                          </Accordion>
                        </div>
                      )}
                    </>
                  );
                })}
                {sourceDocs.length > 0 && (
                  <div className="p-5">
                    <Accordion type="single" collapsible className="flex-col">
                      {sourceDocs.map((doc, index) => (
                        <div key={`SourceDocs-${index}`}>
                          <AccordionItem value={`item-${index}`}>
                            <AccordionTrigger>
                              <h3>
                                {process.env.NEXT_PUBLIC_SOURCE || 'Source'}{' '}
                                {index + 1}
                              </h3>
                            </AccordionTrigger>
                            <AccordionContent>
                              <ReactMarkdown linkTarget="_blank">
                                {doc.pageContent || doc.page_content}
                              </ReactMarkdown>
                            </AccordionContent>
                          </AccordionItem>
                        </div>
                      ))}
                    </Accordion>
                  </div>
                )}
              </div>
            </div>
            <div className={styles.center}>
              <div className={styles.cloudform}>
                <form onSubmit={onSubmit}>
                  <textarea
                    disabled={loading}
                    onKeyDown={handleEnter}
                    ref={textAreaRef}
                    autoFocus={false}
                    rows={1}
                    maxLength={512}
                    id="userInput"
                    name="userInput"
                    placeholder={
                      loading
                        ? process.env.NEXT_PUBLIC_WAITING ||
                        'Waiting for response...'
                        : process.env.NEXT_PUBLIC_QUESTION ||
                        'What is your question?'
                    }
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className={styles.textarea}
                  />
                  <button
                    type="submit"
                    disabled={loading}
                    className={styles.generatebutton}
                  >
                    {loading ? (
                      <div className={styles.loadingwheel}>
                        <LoadingDots color="#000" />
                      </div>
                    ) : (
                      // Send icon SVG in input field
                      <svg
                        viewBox="0 0 20 20"
                        className={styles.svgicon}
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"></path>
                      </svg>
                    )}
                  </button>
                </form>
              </div>
            </div>
            {error && (
              <div className="border border-red-400 rounded-md p-4">
                <p className="text-red-500">{error}</p>
              </div>
            )}
          </main>
        </div>
        <footer className="m-auto p-4">
          <a
            href={
              process.env.NEXT_PUBLIC_FOOTER_LINK || 'https://js.langchain.com'
            }
            target="_blank"
          >
            {process.env.NEXT_PUBLIC_FOOTER1 || 'Powered by LangChain.js.'}
            <br />
            {process.env.NEXT_PUBLIC_FOOTER2 || ''}
          </a>
        </footer>
      </Layout>
    </>
  );
}
