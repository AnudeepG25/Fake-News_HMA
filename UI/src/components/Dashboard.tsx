import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, Info, Loader2, History, LogOut, Plus, X } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
);

interface AnalysisResult {
  id: string;
  isFake: boolean;
  confidence: number;
  reasons: string[];
  text: string;
  created_at: string;
  user_id: string;
}

export default function Dashboard() {
  const [articleText, setArticleText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const { user, signOut } = useAuth();

  useEffect(() => {
    if (user) {
      loadHistory();
    }
  }, [user]);

  const loadHistory = async () => {
    if (!user) return;
    
    const { data, error } = await supabase
      .from('analysis_history')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false });
    
    if (data && !error) {
      setHistory(data);
    }
  };

  const getArticleTitle = (text: string) => {
    const firstLine = text.split('\n')[0].trim();
    return firstLine.length > 60 ? firstLine.substring(0, 57) + '...' : firstLine;
  };

  const startNewAnalysis = () => {
    setArticleText('');
    setResult(null);
  };

  const analyzeArticle = async () => {
    if (!user || !articleText.trim()) return;
    
    setIsAnalyzing(true);
    
    try {
      // Simulated analysis
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const newAnalysis = {
        id: crypto.randomUUID(),
        user_id: user.id,
        text: articleText,
        isFake: articleText.toLowerCase().includes('miracle') || articleText.toLowerCase().includes('shocking'),
        confidence: Math.random() * 30 + 70,
        reasons: [
          'Analysis of language patterns',
          'Source credibility check',
          'Cross-reference with known facts'
        ],
        created_at: new Date().toISOString()
      };
      
      const { error: insertError } = await supabase
        .from('analysis_history')
        .insert([newAnalysis]);

      if (insertError) {
        throw insertError;
      }

      setResult(newAnalysis);
      await loadHistory();
    } catch (error) {
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 flex">
      {/* Sidebar */}
      <div 
        className={`fixed inset-y-0 left-0 transform ${
          isHistoryOpen ? 'translate-x-0' : '-translate-x-full'
        } md:relative md:translate-x-0 transition-transform duration-300 ease-in-out w-80 bg-white border-r border-gray-200 z-20 flex flex-col`}
      >
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">History</h2>
            <button 
              onClick={() => setIsHistoryOpen(false)}
              className="md:hidden text-gray-500 hover:text-gray-700"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>
        
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={startNewAnalysis}
            className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Analysis
          </button>
        </div>

        <div className="overflow-y-auto flex-1">
          {history.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setArticleText(item.text);
                setResult(item);
                if (window.innerWidth < 768) {
                  setIsHistoryOpen(false);
                }
              }}
              className={`w-full text-left p-4 border-b border-gray-100 hover:bg-gray-50 transition-colors ${
                result?.id === item.id ? 'bg-blue-50' : ''
              }`}
            >
              <div className="flex items-center gap-3">
                {item.isFake ? (
                  <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {getArticleTitle(item.text)}
                  </p>
                  <p className="text-xs text-gray-500">
                    {new Date(item.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 min-w-0">
        <div className="p-6">
          <header className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setIsHistoryOpen(true)}
                className="md:hidden text-gray-600 hover:text-gray-900"
              >
                <History className="w-6 h-6" />
              </button>
              <h1 className="text-2xl font-bold text-gray-900">Fact Check AI</h1>
            </div>
            <button
              onClick={signOut}
              className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
            >
              <LogOut className="w-5 h-5" />
              <span>Sign out</span>
            </button>
          </header>

          <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
            <div className="mb-6">
              <label 
                htmlFor="article" 
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Paste your article text
              </label>
              <textarea
                id="article"
                rows={8}
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter the article text you want to analyze..."
                value={articleText}
                onChange={(e) => setArticleText(e.target.value)}
              />
            </div>

            <button
              onClick={analyzeArticle}
              disabled={!articleText.trim() || isAnalyzing}
              className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Info className="w-5 h-5" />
                  Analyze Article
                </>
              )}
            </button>
          </div>

          {result && (
            <div className={`bg-white rounded-xl shadow-lg p-6 ${
              result.isFake ? 'border-l-4 border-red-500' : 'border-l-4 border-green-500'
            }`}>
              <div className="flex items-start gap-4">
                {result.isFake ? (
                  <AlertTriangle className="w-8 h-8 text-red-500 flex-shrink-0" />
                ) : (
                  <CheckCircle className="w-8 h-8 text-green-500 flex-shrink-0" />
                )}
                <div>
                  <h2 className="text-xl font-semibold mb-2">
                    {result.isFake ? 'Potentially Fake News' : 'Likely Authentic'}
                  </h2>
                  <div className="mb-4">
                    <div className="text-sm text-gray-600 mb-1">Confidence Score</div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          result.isFake ? 'bg-red-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${result.confidence}%` }}
                      ></div>
                    </div>
                    <div className="text-right text-sm text-gray-600 mt-1">
                      {result.confidence.toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Analysis Factors:</h3>
                    <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                      {result.reasons.map((reason, index) => (
                        <li key={index}>{reason}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}