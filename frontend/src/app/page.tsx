"use client";

import React, { useState, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import {
  Sparkles, CheckCircle2, XCircle, FileText, Briefcase,
  AlertTriangle, Upload, X, FileType2, BarChart3, CopyPlus,
  Trophy, Users, Filter, ChevronRight, GraduationCap, Clock, Lightbulb
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────
type AppTab = "single" | "batch";
type InputMode = "text" | "file";
interface PanelState {
  mode: InputMode;
  text: string;
  fileName: string;
  wordCount: number;
  isDragging: boolean;
  isLoading: boolean;
  error: string;
}

const initPanel = (): PanelState => ({
  mode: "text", text: "", fileName: "", wordCount: 0,
  isDragging: false, isLoading: false, error: ""
});

// ─── Batch Match Result Type ─────────────────────────────
interface BatchResult {
  id: string; // Tên file
  score: number;
  status: string;
  matched_count: number;
  semantic_sim: number;
}

// ─── Input Panel Component ─────────────────────────────
function InputPanel({
  id, label, icon, accentClass, accentBorder, placeholder, state, onChange, isMultiple = false
}: {
  id: string; label: string; icon: React.ReactNode; accentClass: string; 
  accentBorder: string; placeholder: string; state: PanelState; 
  onChange: (u: Partial<PanelState>) => void; isMultiple?: boolean
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadFiles = async (files: FileList | File[]) => {
    if (isMultiple) {
      // Logic upload nhiều file (Cho Batch Mode)
      onChange({ isLoading: true, error: "" });
      const results = [];
      for (const f of Array.from(files)) {
        if (!["pdf", "docx", "txt"].includes(f.name.split(".").pop()?.toLowerCase() || "")) continue;
        const form = new FormData();
        form.append("file", f);
        try {
          const res = await fetch("http://localhost:5000/api/parse-file", { method: "POST", body: form });
          const data = await res.json();
          results.push({ name: f.name, text: data.text });
        } catch (e) { console.error(e); }
      }
      // Gom tất cả text lại để preview (trong batch mode ta lưu data riêng)
      onChange({ 
        isLoading: false, 
        fileName: `${results.length} files selected`, 
        text: results.map(r => `--- ${r.name} ---\n${r.text.slice(0, 100)}...`).join("\n"),
        mode: "file"
      });
      // Phát sự kiện custom cho batch mode
      const event = new CustomEvent('batchFilesLoaded', { detail: results });
      window.dispatchEvent(event);
    } else {
      // Logic 1 file (Single Mode)
      const f = files[0];
      if (!f) return;
      onChange({ isLoading: true, error: "", fileName: "", text: "" });
      const form = new FormData(); form.append("file", f);
      try {
        const res = await fetch("http://localhost:5000/api/parse-file", { method: "POST", body: form });
        const data = await res.json();
        onChange({ text: data.text, fileName: data.filename, wordCount: data.word_count, mode: "file", isLoading: false });
      } catch (err: any) { onChange({ isLoading: false, error: err.message }); }
    }
  };

  return (
    <div className={`bg-[#0B101A] border rounded-2xl p-6 shadow-xl space-y-4 ${accentBorder} transition-all`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={accentClass}>{icon}</span>
          <h3 className="text-white font-bold text-sm uppercase tracking-wider">{label}</h3>
        </div>
        <button onClick={() => state.mode === "file" ? onChange({ mode: "text", fileName: "", text: "" }) : onChange({ mode: "file" })} className={`text-[10px] px-3 py-1 rounded-full border transition-all font-semibold flex items-center gap-1.5 ${state.mode === "file" ? "border-rose-500/30 text-rose-400" : "border-slate-700 text-slate-400"}`}>
          {state.mode === "file" ? <><X className="w-3 h-3" /> Clear</> : <><Upload className="w-3 h-3" /> {isMultiple ? "Upload Batch" : "Upload File"}</>}
        </button>
      </div>

      {state.mode === "file" && !state.fileName && !state.isLoading && (
        <div onClick={() => fileInputRef.current?.click()} className="flex flex-col items-center justify-center h-44 rounded-xl border-2 border-dashed border-slate-700 hover:border-violet-500 cursor-pointer transition-all">
          <input ref={fileInputRef} type="file" multiple={isMultiple} className="hidden" onChange={(e) => e.target.files && uploadFiles(e.target.files)} />
          <Upload className="w-8 h-8 text-slate-600 mb-2" />
          <p className="text-slate-400 text-xs font-semibold">Click / Drag {isMultiple ? "multiple files" : "one file"}</p>
        </div>
      )}

      {state.isLoading && <div className="h-44 flex items-center justify-center"><div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" /></div>}

      {state.mode === "text" && !state.isLoading && (
        <textarea className="w-full h-44 bg-[#06080D]/50 border border-slate-800 rounded-xl p-4 text-xs text-slate-300 outline-none focus:border-violet-500/40 transition-all resize-none" placeholder={placeholder} value={state.text} onChange={(e) => onChange({ text: e.target.value })} />
      )}

      {state.mode === "file" && state.fileName && !state.isLoading && (
        <div className="bg-[#06080D]/50 border border-slate-800 rounded-xl p-4 h-44 overflow-y-auto text-[10px] text-slate-400 font-mono whitespace-pre-wrap">
          <div className="text-emerald-400 font-bold mb-2">✅ {state.fileName} Loaded</div>
          {state.text}
        </div>
      )}
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────
export default function AI_CV_Matcher_Pro() {
  const [activeTab, setActiveTab] = useState<AppTab>("single");
  const [cvPanel, setCvPanel] = useState(initPanel());
  const [jdPanel, setJdPanel] = useState(initPanel());
  const [analyzing, setAnalyzing] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [batchData, setBatchData] = useState<any[]>([]); // Lưu list text CV cho batch mode

  // Results
  const [resultData, setResultData] = useState<any>(null);
  const [leaderboard, setLeaderboard] = useState<BatchResult[]>([]);
  const [suggestions, setSuggestions] = useState<any>(null);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);

  // Lắng nghe sự kiện load file hàng loạt
  React.useEffect(() => {
    const handler = (e: any) => setBatchData(e.detail);
    window.addEventListener('batchFilesLoaded', handler);
    return () => window.removeEventListener('batchFilesLoaded', handler);
  }, []);

  const handleAnalyze = async () => {
    if (activeTab === "single") {
      if (!cvPanel.text || !jdPanel.text) return alert("Please provide JD and CV");
      setAnalyzing(true); setShowResult(false);
      try {
        const res = await fetch("http://localhost:5000/api/match", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ jd: jdPanel.text, cv: cvPanel.text })
        });
        setResultData(await res.json()); setShowResult(true);
      } catch (e) { alert("Server Error"); } finally { setAnalyzing(false); }
    } else {
      // BATCH MODE
      let finalBatch = batchData;
      
      // Nếu người dùng không upload file mà lại dán text vào textarea
      if (finalBatch.length === 0 && cvPanel.text.trim()) {
        // Tách các CV dựa trên dấu hiệu phân cách (dòng bắt đầu bằng CV_ hoặc --- hoặc 2 dấu xuống dòng)
        const parts = cvPanel.text.split(/\n(?=CV_)|---|\n\n\n/).filter(p => p.trim());
        finalBatch = parts.map((p, i) => ({
          name: p.split('\n')[0].substring(0, 30) || `Pasted CV ${i + 1}`,
          text: p
        }));
      }

      if (!jdPanel.text || finalBatch.length === 0) return alert("Please provide JD and multiple CVs (Upload files or Paste with separators)");
      
      setAnalyzing(true); setShowResult(false);
      try {
        const res = await fetch("http://localhost:5000/api/batch-match", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            jd: jdPanel.text, 
            cvs: finalBatch.map(b => ({ id: b.name, text: b.text })) 
          })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        setLeaderboard(data.leaderboard); setShowResult(true);
      } catch (e: any) { alert("Batch API Error: " + e.message); } finally { setAnalyzing(false); }
    }
  };

  const getStatusColor = (s: string) => {
    if (s === "EXCELLENT") return "text-emerald-400 bg-emerald-400/10 border-emerald-400/20";
    if (s === "POTENTIAL") return "text-violet-400 bg-violet-400/10 border-violet-400/20";
    return "text-slate-400 bg-slate-800 border-slate-700";
  };

  const handleGetSuggestions = async () => {
    if (!cvPanel.text || !jdPanel.text) return alert("Please provide both CV and JD");
    setLoadingSuggestions(true);
    try {
      const res = await fetch("http://localhost:5000/api/suggest-cv-improvements", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jd: jdPanel.text, cv: cvPanel.text })
      });
      setSuggestions(await res.json());
    } catch (e) {
      alert("Error fetching suggestions");
    } finally {
      setLoadingSuggestions(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#06080D] text-slate-300 font-sans overflow-hidden">
      {/* SIDEBAR */}
      <aside className="w-64 border-r border-slate-800/60 bg-[#0A0D14] p-6 flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-3 mb-10">
            <div className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center"><Sparkles className="w-5 h-5 text-white" /></div>
            <div><h2 className="text-white font-bold text-lg">MatchAI Pro</h2><p className="text-[10px] text-slate-500 tracking-widest">BATCH VERSION</p></div>
          </div>
          <nav className="space-y-1">
            <button onClick={() => { setActiveTab("single"); setShowResult(false); }} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === "single" ? "bg-violet-500/10 text-violet-400 border border-violet-500/20" : "text-slate-500 hover:text-slate-200"}`}><CopyPlus className="w-4 h-4" /> 1-vs-1 Match</button>
            <button onClick={() => { setActiveTab("batch"); setShowResult(false); }} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === "batch" ? "bg-violet-500/10 text-violet-400 border border-violet-500/20" : "text-slate-500 hover:text-slate-200"}`}><Trophy className="w-4 h-4" /> Leaderboard</button>
          </nav>
        </div>
        <div className="p-4 bg-slate-800/20 rounded-xl border border-slate-700/50"><p className="text-[10px] text-slate-500 italic">"Ranking 50 CVs in seconds using SBERT local embedding."</p></div>
      </aside>

      {/* MAIN */}
      <div className="flex-1 flex flex-col overflow-hidden relative">
        <header className="h-16 border-b border-slate-800/60 flex items-center px-8 bg-[#0A0D14]/80 backdrop-blur-md">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-widest">{activeTab === "single" ? "Single Pipeline" : "Batch Intelligence Leaderboard"}</div>
        </header>

        <main className="flex-1 overflow-y-auto pt-8 pb-12 px-10">
          <div className="max-w-5xl mx-auto space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <InputPanel id="cv" label={activeTab === "single" ? "Candidate CV" : "Multiple CVs"} icon={<Users className="w-5 h-5" />} accentClass="text-violet-400" accentBorder="border-slate-800/60" placeholder="Paste single CV..." state={cvPanel} onChange={(updates) => setCvPanel(prev => ({ ...prev, ...updates }))} isMultiple={activeTab === "batch"} />
              <InputPanel id="jd" label="Job Description" icon={<Briefcase className="w-5 h-5" />} accentClass="text-indigo-400" accentBorder="border-slate-800/60" placeholder="Paste JD..." state={jdPanel} onChange={(updates) => setJdPanel(prev => ({ ...prev, ...updates }))} />            </div>

            <div className="flex justify-center">
              <button onClick={handleAnalyze} disabled={analyzing} className="bg-violet-600 hover:bg-violet-500 text-white shadow-xl px-12 py-3.5 rounded-full flex items-center gap-3 transition-all font-bold disabled:opacity-50">
                {analyzing ? <><Clock className="w-5 h-5 animate-spin" /> Computing Rank...</> : <><Sparkles className="w-5 h-5" /> START {activeTab === "single" ? "MATCH" : "BATCH RANKING"}</>}
              </button>
            </div>

            {showResult && activeTab === "single" && resultData && (
              <div className="bg-[#0B101A] border border-slate-800/80 rounded-2xl p-8 shadow-2xl animate-in fade-in slide-in-from-bottom-5">
                 <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-8">
                       <div className="relative w-24 h-24">
                          <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                              <circle cx="50" cy="50" r="45" fill="none" stroke="#1E293B" strokeWidth="8" />
                              <circle cx="50" cy="50" r="45" fill="none" stroke="url(#g)" strokeWidth="8" strokeDasharray="282.7" strokeDashoffset={282.7 * (1 - resultData.score / 100)} className="transition-all duration-1000" />
                              <defs><linearGradient id="g"><stop offset="0%" stopColor="#8b5cf6" /><stop offset="100%" stopColor="#10b981" /></linearGradient></defs>
                          </svg>
                          <div className="absolute inset-0 flex items-center justify-center font-black text-white text-xl">{resultData.score}%</div>
                       </div>
                       <div>
                          <div className={`px-4 py-1 rounded-full text-[10px] font-bold border ${getStatusColor(resultData.status)}`}>{resultData.status}</div>
                          <div className="text-xs text-slate-500 mt-2">Semantic Similarity: {resultData.analysis.semantic_sim}</div>
                       </div>
                    </div>
                    <div className="max-w-md bg-[#121124] p-4 rounded-xl border border-violet-500/10 text-xs italic text-slate-300">
                       <ReactMarkdown>{resultData.explanation}</ReactMarkdown>
                    </div>
                 </div>

                 {/* CV IMPROVEMENT SUGGESTIONS SECTION */}
                 <div className="mt-8 pt-8 border-t border-slate-700">
                   <button
                     onClick={handleGetSuggestions}
                     disabled={loadingSuggestions}
                     className="mb-6 bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2 rounded-lg flex items-center gap-2 transition-all font-semibold disabled:opacity-50 text-sm"
                   >
                     {loadingSuggestions ? (
                       <><Clock className="w-4 h-4 animate-spin" /> Generating Suggestions...</>
                     ) : (
                       <><Sparkles className="w-4 h-4" /> Get CV Improvement Tips</>
                     )}
                   </button>

                   {suggestions && !suggestions.error && (
                     <div className="space-y-6">
                       {/* ANALYSIS SUMMARY */}
                       <div className="grid grid-cols-4 gap-4">
                         <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4 text-center border-l-2 border-l-emerald-500">
                           <div className="text-lg font-bold text-emerald-400">{suggestions.analysis.matched_skills?.length || 0}</div>
                           <div className="text-[10px] text-slate-500 mt-1">Matched Skills</div>
                         </div>
                         <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4 text-center border-l-2 border-l-amber-500">
                           <div className="text-lg font-bold text-amber-400">{suggestions.analysis.missing_skills?.length || 0}</div>
                           <div className="text-[10px] text-slate-500 mt-1">Missing Skills</div>
                         </div>
                         <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4 text-center border-l-2 border-l-red-500">
                           <div className="text-lg font-bold text-red-400">{suggestions.analysis.critical_missing?.length || 0}</div>
                           <div className="text-[10px] text-slate-500 mt-1">Critical Missing</div>
                         </div>
                         <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4 text-center border-l-2 border-l-violet-500">
                           <div className="text-lg font-bold text-violet-400">{suggestions.analysis.match_percentage || 0}%</div>
                           <div className="text-[10px] text-slate-500 mt-1">Match Percentage</div>
                         </div>
                       </div>

                       {/* ACTION STEPS */}
                       {suggestions.action_steps && (
                         <div className="bg-slate-900/50 border border-slate-700/30 rounded-lg p-4">
                           <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                             <CheckCircle2 className="w-4 h-4 text-emerald-400" /> Action Steps
                           </h4>
                           <ol className="space-y-2 text-xs text-slate-300">
                             {suggestions.action_steps.map((step: string, i: number) => (
                               <li key={i} className="text-slate-400">
                                 <ReactMarkdown>{step}</ReactMarkdown>
                               </li>
                             ))}
                           </ol>
                         </div>
                       )}

                       {/* IMPROVEMENT SUGGESTIONS */}
                       <div className="space-y-3">
                         <h4 className="text-sm font-bold text-white flex items-center gap-2">
                           <Sparkles className="w-4 h-4 text-amber-400" /> Detailed Suggestions
                         </h4>
                         {suggestions.suggestions?.map((sugg: any, i: number) => (
                           <div
                             key={i}
                             className={`border-l-4 rounded-lg p-4 bg-slate-900/30 transition-all ${
                               sugg.priority === "HIGH"
                                 ? "border-l-red-500 border-red-500/10"
                                 : "border-l-amber-500 border-amber-500/10"
                             }`}
                           >
                             <div className="flex items-start justify-between mb-2">
                               <h5 className="font-bold text-white text-sm flex items-center gap-2">
                                 {sugg.title}
                               </h5>
                               <span className={`px-2 py-0.5 rounded text-[9px] font-bold ${
                                 sugg.priority === "HIGH"
                                   ? "bg-red-500/20 text-red-300"
                                   : "bg-amber-500/20 text-amber-300"
                               }`}>
                                 {sugg.priority}
                               </span>
                             </div>
                             <p className="text-xs text-slate-400 mb-2">{sugg.description}</p>
                             <div className="bg-[#06080D] rounded p-2 mb-2 border border-slate-800">
                               <p className="text-[10px] font-mono text-emerald-400 mb-1">
                                 <strong>Action:</strong>
                               </p>
                               <p className="text-[10px] text-slate-300">{sugg.action}</p>
                             </div>
                             {sugg.example && (
                               <div className="text-[10px] text-slate-500 italic">
                                 <strong>Example:</strong> {sugg.example}
                               </div>
                             )}
                           </div>
                         ))}
                       </div>

                       {/* IMPROVED SUMMARY */}
                       {suggestions.improved_summary && (
                         <div className="bg-slate-900/50 border border-violet-500/20 rounded-lg p-4">
                           <h4 className="text-sm font-bold text-violet-400 mb-3">📝 Improved CV Summary</h4>
                           <div className="bg-[#06080D] rounded p-3 text-xs text-slate-300 max-h-64 overflow-y-auto font-mono">
                             <ReactMarkdown>{suggestions.improved_summary}</ReactMarkdown>
                           </div>
                         </div>
                       )}
                     </div>
                   )}
                 </div>
              </div>
            )}

            {showResult && activeTab === "batch" && leaderboard.length > 0 && (
              <div className="bg-[#0B101A] border border-slate-800/80 rounded-2xl overflow-hidden shadow-2xl animate-in zoom-in-95 duration-500">
                <div className="bg-slate-900/50 p-4 border-b border-slate-800 flex justify-between items-center">
                   <h3 className="text-sm font-bold text-white flex items-center gap-2"><Trophy className="w-4 h-4 text-amber-500" /> AI Leaderboard Ranking</h3>
                   <div className="text-[10px] text-slate-500">Sorted by Hybrid Semantic Score</div>
                </div>
                <table className="w-full text-left">
                   <thead className="bg-[#06080D] text-[10px] uppercase font-black text-slate-500 border-b border-slate-800">
                      <tr>
                        <th className="px-6 py-4">Rank</th>
                        <th className="px-6 py-4">Candidate File</th>
                        <th className="px-6 py-4">Status</th>
                        <th className="px-6 py-4">Skill Match</th>
                        <th className="px-6 py-4 text-right">Match Score</th>
                      </tr>
                   </thead>
                   <tbody className="divide-y divide-slate-800/50">
                      {leaderboard.map((res, i) => (
                        <tr key={res.id} className={`hover:bg-slate-800/30 transition-colors ${i === 0 ? "bg-violet-500/5" : ""}`}>
                           <td className="px-6 py-4 font-black">
                              {i === 0 ? <span className="text-amber-500 text-lg">🥇</span> : i + 1}
                           </td>
                           <td className="px-6 py-4 min-w-[200px]">
                              <div className="flex flex-col">
                                 <span className="text-xs font-bold text-white truncate max-w-[200px]">{res.id}</span>
                                 <span className="text-[10px] text-slate-600">Vector Sim: {res.semantic_sim}</span>
                              </div>
                           </td>
                           <td className="px-6 py-4">
                              <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold border ${getStatusColor(res.status)}`}>{res.status}</span>
                           </td>
                           <td className="px-6 py-4 text-emerald-500 font-bold text-xs">{res.matched_count} points</td>
                           <td className="px-6 py-4 text-right">
                              <div className="flex flex-col items-end">
                                 <span className="text-lg font-black text-white">{res.score}%</span>
                                 <div className="w-20 h-1 bg-slate-800 rounded-full mt-1 overflow-hidden">
                                    <div className="h-full bg-gradient-to-r from-violet-500 to-emerald-500" style={{ width: `${res.score}%` }}></div>
                                 </div>
                              </div>
                           </td>
                        </tr>
                      ))}
                   </tbody>
                </table>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
