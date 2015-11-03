/** \file tb2regularflowconstr.hpp
 *  \brief Flow based global cost function : sregular_flow
 */

#ifndef TB2REGULARFLOWCONSTR_HPP_
#define TB2REGULARFLOWCONSTR_HPP_

#include "tb2flowbasedconstr.hpp"
#include "tb2automaton.hpp"

class RegularFlowConstraint : public FlowBasedGlobalConstraint {
private:

    struct DFA : public WeightedAutomaton {
        vector<int> init;
        vector<int> final;
        set<int> symbol;
        vector<pair<int, int> > *transition;
        int nstate;

        DFA() : transition(NULL), nstate(0) {}
        
        void setNumStates(int size) {
            transition = new vector<pair<int, int> >[size];
            nstate = size;
        }
        
        void addInitialState(int begin) {
            init.push_back(begin);
        }
        
        void addFinalState(int end) {
            final.push_back(end);
        }

        int size() {
            return nstate;
        }

        void addTransition(int start, int ch, int end, int weight) {
            transition[start].push_back(make_pair(ch, end));
        }

        int getNextState(int start, int ch) {
            int next = -1;
            for (vector<pair<int, int> >::iterator i = transition[start].begin();
                    i != transition[start].end(); i++) {
                if (i->first == ch) next = i->second;
            }
            return next;
        }

        set<int> getSymbolNeed(int start, int end) {
            set<int> require;
            for (vector<pair<int, int> >::iterator i = transition[start].begin();
                    i != transition[start].end(); i++) {
                if (i->second == end) require.insert(i->first);
            }
            return require;
        }

        void print() {
            cout << "start state : ";
            for (vector<int>::iterator i = init.begin(); i != init.end(); i++) cout << *i << " ";
            cout << endl;
            for (int s = 0; s < nstate; s++) {
                for (vector<pair<int, int> >::iterator i = transition[s].begin(); i != transition[s].end(); i++)
                    cout << s << " -" << i->first << "-> " << i->second << endl;
            }
            cout << "end state : ";
            for (vector<int>::iterator i = final.begin(); i != final.end(); i++) cout << *i << " ";
            cout << endl;
        }
    };
    
    template<class Element> 
    struct min_priority_queue: public priority_queue<Element, vector<Element>, greater<Element> > {};
        
    static const int EDIT = 1;
    static const int VAR = 0;
    
    static const int INS_TAG = -(INT_MAX >> 3);


    int subdef, insdef, deldef;
    DFA dfa;
    typedef vector<map<int, map<int, Cost> > > CostTable; //[start][char][end]
    CostTable costTb;
    int epsilonChar;
    
    vector<Cost> fromSource;
    vector<Cost> toSink;
    vector<vector<Cost> > table;
    vector<set<int> > tempdomain;
    vector<set<int> > predomain;
    vector<set<int> > curdomain;
    vector<vector<vector<pair<int, int> > > > mapedge;

    pair<int, int> mapto(int varindex, Value val) {
        return make_pair(0, 0);
    }

    Cost constructFlow(Graph &graph) {        
        computeShortestPath(graph, cost);
        return cost;
    }
    void checkRemoved(Graph &graph, StoreCost &cost, vector<int> &rmv);
    void augmentStructure(Graph &graph, StoreCost &cost, int varindex, map<Value, Cost> &delta);
    void findProjection(Graph &graph, StoreCost &cost, int varindex, map<Value, Cost> &delta);
    size_t GetGraphAllocatedSize();
    void buildGraph(Graph &g);
    void buildGraphBasic(Graph &g, bool needRebuildIndex);
    void computeShortestPath(Graph &g, StoreCost &cost);
    
    void buildWeightedDFATable();

    /*Cost evalOriginal(String s) {
        return 0;
    }*/
    Cost evalOriginal(String s);
            
    
public:
    RegularFlowConstraint(WCSP *wcsp, EnumeratedVariable** scope_in, int arity_in);

    ~RegularFlowConstraint() {
    }

    string getName() {
        return "regular constraint";
    }
    //Cost eval(String s);
    void read(istream &file);
    WeightedAutomaton* getWeightedAutomaton() {return &dfa;}
    void organizeConfig();

    virtual Cost getMinCost() {
        return constructFlow(*graph);     
    }    

    void print(ostream& os);
    
};

#endif
