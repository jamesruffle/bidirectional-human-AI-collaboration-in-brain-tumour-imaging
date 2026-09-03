import React from 'react';
import { Car, Plane, User, Cpu, Shield, Eye } from 'lucide-react';

export default function HumanAIParadigm() {
  return (
    <div className="w-full max-w-6xl mx-auto p-8 bg-white">
      <h1 className="text-3xl font-bold text-center mb-12 text-slate-800">
        Human-AI Collaboration Paradigms
      </h1>
      
      <div className="grid md:grid-cols-2 gap-8">
        {/* Left Column - AI-Assisted Human Agent */}
        <div className="space-y-6">
          {/* Car Lane Assist Panel */}
          <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-blue-200">
            <div className="flex items-center justify-center mb-4">
              <div className="bg-blue-100 p-4 rounded-full">
                <Car className="w-12 h-12 text-blue-600" />
              </div>
            </div>
            
            <h2 className="text-2xl font-bold text-center mb-2 text-blue-900">
              a) AI-Assisted Human Agent
            </h2>
            <p className="text-center text-lg font-semibold text-blue-700 mb-6">
              Car Lane Assist
            </p>
            
            <div className="space-y-6">
              {/* Visual Flow */}
              <div className="flex items-center justify-center gap-4">
                <div className="flex flex-col items-center">
                  <div className="bg-blue-600 text-white p-4 rounded-lg shadow-md flex items-center gap-2 w-40 justify-center">
                    <User className="w-5 h-5" />
                    <span className="font-semibold">Human</span>
                  </div>
                  <p className="text-xs text-slate-600 mt-2 text-center">Driver</p>
                </div>
                
                <div className="flex flex-col items-center">
                  <div className="text-3xl text-blue-600">→</div>
                  <div className="text-xs text-slate-500 mt-1">controls</div>
                </div>
                
                <div className="flex flex-col items-center">
                  <div className="bg-slate-700 text-white p-4 rounded-lg shadow-md flex items-center gap-2 w-40 justify-center">
                    <Car className="w-5 h-5" />
                    <span className="font-semibold">Task</span>
                  </div>
                  <p className="text-xs text-slate-600 mt-2 text-center">Driving</p>
                </div>
              </div>
              
              {/* AI Support Arrow */}
              <div className="flex items-center justify-center">
                <div className="flex items-center gap-3 bg-green-50 border-2 border-green-300 rounded-lg p-3">
                  <Shield className="w-6 h-6 text-green-600" />
                  <div className="text-sm">
                    <div className="flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-green-600" />
                      <span className="font-semibold text-green-800">AI Agent</span>
                    </div>
                    <p className="text-xs text-green-700">Provides safety assistance</p>
                  </div>
                </div>
              </div>
              
              {/* Description */}
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <ul className="space-y-2 text-sm text-slate-700">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 font-bold mt-1">•</span>
                    <span><strong>Human is in control:</strong> Driver steers, accelerates, brakes</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 font-bold mt-1">•</span>
                    <span><strong>AI assists:</strong> Monitors lane position, alerts, gentle corrections</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 font-bold mt-1">•</span>
                    <span><strong>Human responsibility:</strong> Driver accountable for all outcomes</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 font-bold mt-1">•</span>
                    <span><strong>AI role:</strong> Safety net and enhancement</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Human-Assisted AI Agent */}
        <div className="space-y-6">
          {/* Plane Autopilot Panel */}
          <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-purple-200">
            <div className="flex items-center justify-center mb-4">
              <div className="bg-purple-100 p-4 rounded-full">
                <Plane className="w-12 h-12 text-purple-600" />
              </div>
            </div>
            
            <h2 className="text-2xl font-bold text-center mb-2 text-purple-900">
              b) Human-Assisted AI Agent
            </h2>
            <p className="text-center text-lg font-semibold text-purple-700 mb-6">
              Plane Autopilot
            </p>
            
            <div className="space-y-6">
              {/* Visual Flow */}
              <div className="flex items-center justify-center gap-4">
                <div className="flex flex-col items-center">
                  <div className="bg-purple-600 text-white p-4 rounded-lg shadow-md flex items-center gap-2 w-40 justify-center">
                    <Cpu className="w-5 h-5" />
                    <span className="font-semibold">AI Agent</span>
                  </div>
                  <p className="text-xs text-slate-600 mt-2 text-center">Autopilot</p>
                </div>
                
                <div className="flex flex-col items-center">
                  <div className="text-3xl text-purple-600">→</div>
                  <div className="text-xs text-slate-500 mt-1">controls</div>
                </div>
                
                <div className="flex flex-col items-center">
                  <div className="bg-slate-700 text-white p-4 rounded-lg shadow-md flex items-center gap-2 w-40 justify-center">
                    <Plane className="w-5 h-5" />
                    <span className="font-semibold">Task</span>
                  </div>
                  <p className="text-xs text-slate-600 mt-2 text-center">Flying</p>
                </div>
              </div>
              
              {/* Human Support Arrow */}
              <div className="flex items-center justify-center">
                <div className="flex items-center gap-3 bg-amber-50 border-2 border-amber-300 rounded-lg p-3">
                  <Eye className="w-6 h-6 text-amber-600" />
                  <div className="text-sm">
                    <div className="flex items-center gap-2">
                      <User className="w-4 h-4 text-amber-600" />
                      <span className="font-semibold text-amber-800">Human Agent</span>
                    </div>
                    <p className="text-xs text-amber-700">Monitors and intervenes</p>
                  </div>
                </div>
              </div>
              
              {/* Description */}
              <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                <ul className="space-y-2 text-sm text-slate-700">
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 font-bold mt-1">•</span>
                    <span><strong>AI is in control:</strong> Autopilot maintains altitude, heading, speed</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 font-bold mt-1">•</span>
                    <span><strong>Human monitors:</strong> Pilot oversees systems, ready to intervene</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 font-bold mt-1">•</span>
                    <span><strong>AI responsibility:</strong> System handles routine operations</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 font-bold mt-1">•</span>
                    <span><strong>Human role:</strong> Strategic oversight and edge cases</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
