#!/usr/bin/env python3
"""
Ajouter des logs VISIBLES qui montrent le modèle utilisé
"""
import re

def add_visible_model_logs():
    file_path = '/app/archon_graph.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ajouter des logs VISIBLES au début d'invoke_agent
    if 'def invoke_agent(thread_id: str, user_input: str):' in content:
        old_func_start = '''def invoke_agent(thread_id: str, user_input: str):
    """Invoque l'agent avec logs détaillés"""
    logger.info("="*80)
    logger.info(f"🚀 DÉMARRAGE INVOKE_AGENT - Thread: {thread_id}")
    logger.info(f"🚀 Message utilisateur: {user_input}")
    logger.info(f"🚀 Profil actuel: {llm_provider.config.provider} / {llm_provider.config.primary_model}")
    logger.info("="*80)'''
        
        new_func_start = '''def invoke_agent(thread_id: str, user_input: str):
    """Invoque l'agent avec logs détaillés"""
    print("="*80)
    print(f"🚀 ARCHON START - Thread: {thread_id}")
    print(f"🚀 Message: {user_input}")
    print(f"🚀 Provider: {llm_provider.config.provider}")
    print(f"🚀 Model: {llm_provider.config.primary_model}")
    print(f"🚀 Model Reasoner: {llm_provider.config.reasoner_model}")
    print("="*80)
    
    logger.info("="*80)
    logger.info(f"🚀 DÉMARRAGE INVOKE_AGENT - Thread: {thread_id}")
    logger.info(f"🚀 Message utilisateur: {user_input}")
    logger.info(f"🚀 Profil actuel: {llm_provider.config.provider} / {llm_provider.config.primary_model}")
    logger.info("="*80)'''
        
        content = content.replace(old_func_start, new_func_start)
    
    # Ajouter des prints dans define_scope_with_reasoner
    content = content.replace(
        'def define_scope_with_reasoner(state: AgentState) -> AgentState:',
        '''def define_scope_with_reasoner(state: AgentState) -> AgentState:
    """Définit la portée avec l'agent reasoner"""
    print("🔍 REASONER - Starting with model:", llm_provider.config.reasoner_model)
    logger.info("="*50)
    logger.info("🔍 REASONER STARTING")
    logger.info(f"🔍 Modèle: {llm_provider.config.reasoner_model}")
    logger.info("="*50)'''
    )
    
    # Ajouter des prints dans advisor_with_examples
    content = content.replace(
        'def advisor_with_examples(state: AgentState) -> AgentState:',
        '''def advisor_with_examples(state: AgentState) -> AgentState:
    """Génère des conseils avec l'agent advisor"""
    print("💡 ADVISOR - Starting with model:", llm_provider.config.primary_model)
    logger.info("="*50)
    logger.info("💡 ADVISOR STARTING")
    logger.info(f"💡 Modèle: {llm_provider.config.primary_model}")
    logger.info("="*50)'''
    )
    
    # Ajouter des prints dans coder_agent
    content = content.replace(
        'def coder_agent(state: AgentState) -> AgentState:',
        '''def coder_agent(state: AgentState) -> AgentState:
    """Génère le code avec l'agent coder"""
    print("⚡ CODER - Starting with model:", llm_provider.config.primary_model)
    logger.info("="*50)
    logger.info("⚡ CODER STARTING")
    logger.info(f"⚡ Modèle: {llm_provider.config.primary_model}")
    logger.info("="*50)'''
    )
    
    # Écrire le fichier corrigé
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Logs VISIBLES ajoutés avec print() dans archon_graph.py")
    print("  - Logs au démarrage d'invoke_agent")
    print("  - Logs dans chaque fonction d'agent")
    print("  - Affichage des modèles utilisés")

if __name__ == "__main__":
    add_visible_model_logs()
