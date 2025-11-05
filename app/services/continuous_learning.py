"""
Continuous Learning Engine - DARWIN Neural Extension
Sistema de ML automático que aprende com suas interações em todas as plataformas

Objetivo: Transformar DARWIN em extensão digital do seu cérebro
- Aprende com cada conversa (ChatGPT, Claude, Gemini, Copilot)
- Identifica seus padrões de pensamento
- Antecipa suas necessidades de pesquisa
- Personaliza recomendações
- Conecta conhecimentos cross-domain como você faz

Features:
- User behavior modeling
- Topic preference learning
- Interaction pattern recognition
- Personalized embedding space
- Adaptive search ranking
- Proactive knowledge suggestions
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class UserInteraction:
    """Representa uma interação do usuário"""
    timestamp: datetime
    platform: str  # chatgpt, claude, gemini, copilot, claude-code
    domain: str  # biomaterials, chemistry, research, etc.
    query: str
    memory_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Metadata de comportamento
    search_performed: bool = False
    memory_saved: bool = False
    follow_up_count: int = 0
    session_duration_seconds: float = 0.0
    
    # Feedback implícito
    result_clicked: Optional[int] = None  # Posição do resultado clicado
    time_on_result_seconds: float = 0.0
    copied_content: bool = False
    created_follow_up: bool = False


@dataclass
class UserProfile:
    """Perfil do usuário com padrões aprendidos"""
    user_id: str = "agourakis"  # Você é o único usuário!
    
    # Domínios de interesse (aprendido)
    domain_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Plataformas favoritas por tipo de tarefa
    platform_by_task: Dict[str, str] = field(default_factory=dict)
    
    # Padrões temporais
    active_hours: List[int] = field(default_factory=list)
    typical_session_duration: float = 0.0
    
    # Conhecimento acumulado
    expertise_areas: List[str] = field(default_factory=list)
    learning_areas: List[str] = field(default_factory=list)
    
    # Embedding personalizado (espaço vetorial único para você!)
    personal_embedding_bias: Optional[np.ndarray] = None
    
    # Conexões conceituais que VOCÊ faz
    cross_domain_links: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Última atualização
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ContinuousLearningEngine:
    """
    Motor de aprendizado contínuo - Core do DARWIN Neural Extension
    
    Aprende:
    1. Seus tópicos de interesse ao longo do tempo
    2. Padrões de busca e pesquisa
    3. Conexões cross-domain que VOCÊ faz
    4. Estilo de escrita e pensamento
    5. Necessidades de pesquisa antes de você pedir
    """
    
    def __init__(
        self,
        semantic_memory_service,
        min_interactions_for_training: int = 50,
        retrain_interval_hours: int = 24,
    ):
        self.semantic_memory = semantic_memory_service
        self.min_interactions = min_interactions_for_training
        self.retrain_interval = retrain_interval_hours
        
        # Histórico de interações
        self.interaction_history: List[UserInteraction] = []
        
        # Perfil do usuário (VOCÊ!)
        self.user_profile = UserProfile()
        
        # Modelos ML
        self.preference_model: Optional[GradientBoostingRegressor] = None
        self.topic_classifier: Optional[RandomForestClassifier] = None
        self.clustering_model: Optional[DBSCAN] = None
        
        # Estado de treinamento
        self.last_training_time: Optional[datetime] = None
        self.is_training: bool = False
        
        # Métricas
        self.metrics = {
            "total_interactions": 0,
            "trainings_performed": 0,
            "prediction_accuracy": 0.0,
            "user_satisfaction_score": 0.0,
        }
        
        logger.info("🧠 Continuous Learning Engine initialized for user: agourakis")
    
    async def record_interaction(self, interaction: UserInteraction):
        """
        Registra uma interação do usuário
        
        Chamado automaticamente quando:
        - Você salva uma memória via MCP
        - Busca algo no DARWIN
        - Usa Custom GPT
        - Conversa em qualquer plataforma
        """
        self.interaction_history.append(interaction)
        self.metrics["total_interactions"] += 1
        
        logger.info(
            f"📝 Interaction recorded | platform={interaction.platform} | "
            f"domain={interaction.domain} | total={len(self.interaction_history)}"
        )
        
        # Atualiza perfil incrementalmente
        await self._update_profile_incremental(interaction)
        
        # Verifica se deve re-treinar
        should_retrain = await self._should_retrain()
        if should_retrain:
            asyncio.create_task(self._retrain_models())
    
    async def _update_profile_incremental(self, interaction: UserInteraction):
        """Atualiza perfil do usuário incrementalmente (sem esperar re-treino)"""
        
        # 1. Atualiza preferências de domínio
        domain = interaction.domain
        current_score = self.user_profile.domain_preferences.get(domain, 0.0)
        
        # Decay das preferências antigas + boost da nova
        for d in self.user_profile.domain_preferences:
            self.user_profile.domain_preferences[d] *= 0.99  # Decay 1%
        
        # Incrementa preferência atual
        weight = 1.0
        if interaction.memory_saved:
            weight = 2.0  # Se salvou memória, peso maior!
        if interaction.follow_up_count > 0:
            weight *= (1 + interaction.follow_up_count * 0.1)  # Follow-ups = interesse
        
        self.user_profile.domain_preferences[domain] = current_score + weight
        
        # 2. Atualiza horários ativos
        hour = interaction.timestamp.hour
        if hour not in self.user_profile.active_hours:
            self.user_profile.active_hours.append(hour)
        
        # 3. Aprende plataforma preferida por domínio
        key = f"{domain}_platform"
        platform_counts = getattr(self, '_platform_counts', defaultdict(Counter))
        platform_counts[domain][interaction.platform] += 1
        
        # Plataforma mais usada para esse domínio
        most_used = platform_counts[domain].most_common(1)
        if most_used:
            self.user_profile.platform_by_task[domain] = most_used[0][0]
        
        self._platform_counts = platform_counts
        
        # 4. Detecta conexões cross-domain
        await self._detect_cross_domain_links(interaction)
        
        self.user_profile.last_updated = datetime.now(timezone.utc)
    
    async def _detect_cross_domain_links(self, interaction: UserInteraction):
        """
        Detecta quando você conecta conceitos de diferentes domínios
        
        Ex: Se você busca "biomaterials" e depois "chemistry" na mesma sessão,
        DARWIN aprende que você frequentemente conecta esses domínios
        """
        # Pega interações recentes (última hora)
        recent = [
            i for i in self.interaction_history[-20:]
            if (interaction.timestamp - i.timestamp).seconds < 3600
        ]
        
        if len(recent) < 2:
            return
        
        # Identifica transições de domínio
        current_domain = interaction.domain
        for prev_interaction in recent:
            prev_domain = prev_interaction.domain
            
            if prev_domain != current_domain:
                # Você conectou dois domínios!
                link = tuple(sorted([prev_domain, current_domain]))
                
                current_strength = self.user_profile.cross_domain_links.get(link, 0.0)
                self.user_profile.cross_domain_links[link] = current_strength + 1.0
                
                logger.info(f"🔗 Cross-domain link detected: {prev_domain} ↔ {current_domain}")
    
    async def _should_retrain(self) -> bool:
        """Decide se deve re-treinar modelos"""
        
        # Precisa de mínimo de interações
        if len(self.interaction_history) < self.min_interactions:
            return False
        
        # Já está treinando
        if self.is_training:
            return False
        
        # Treina a cada X horas
        if self.last_training_time:
            hours_since_training = (
                datetime.now(timezone.utc) - self.last_training_time
            ).total_seconds() / 3600
            
            if hours_since_training < self.retrain_interval:
                return False
        
        return True
    
    async def _retrain_models(self):
        """
        Re-treina modelos ML com histórico de interações
        
        Modelos treinados:
        1. Preference Model: Prediz relevância de conteúdo para você
        2. Topic Classifier: Classifica automaticamente domínio de queries
        3. Clustering: Agrupa seus tópicos de pesquisa
        """
        if self.is_training:
            return
        
        self.is_training = True
        logger.info("🚀 Starting model retraining...")
        
        try:
            # 1. Prepara dataset
            X, y_relevance, y_domain = await self._prepare_training_data()
            
            if len(X) < 20:
                logger.warning("Not enough data for training")
                return
            
            # 2. Treina modelo de preferências
            self.preference_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.preference_model.fit(X, y_relevance)
            
            # 3. Treina classificador de tópicos
            self.topic_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.topic_classifier.fit(X, y_domain)
            
            # 4. Clustering de interesses
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
            clusters = self.clustering_model.fit_predict(X_scaled)
            
            # 5. Identifica áreas de expertise vs learning
            await self._identify_expertise_areas(clusters)
            
            # 6. Cria embedding personalizado
            await self._create_personal_embedding()
            
            self.last_training_time = datetime.now(timezone.utc)
            self.metrics["trainings_performed"] += 1
            
            logger.info(
                f"✅ Models retrained successfully | "
                f"samples={len(X)} | clusters={len(set(clusters))}"
            )
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
        finally:
            self.is_training = False
    
    async def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepara dados de treinamento do histórico"""
        
        X = []
        y_relevance = []
        y_domain = []
        
        for interaction in self.interaction_history:
            # Features
            features = [
                # Preferência de domínio atual
                self.user_profile.domain_preferences.get(interaction.domain, 0.0),
                
                # Hora do dia (normalizada)
                interaction.timestamp.hour / 24.0,
                
                # Dia da semana
                interaction.timestamp.weekday() / 7.0,
                
                # Duração da sessão (normalizada)
                min(interaction.session_duration_seconds / 3600, 1.0),
                
                # Plataforma (one-hot encoded)
                1.0 if interaction.platform == "chatgpt" else 0.0,
                1.0 if interaction.platform == "claude" else 0.0,
                1.0 if interaction.platform == "claude-code" else 0.0,
                1.0 if interaction.platform == "gemini" else 0.0,
                
                # Comportamento
                1.0 if interaction.memory_saved else 0.0,
                1.0 if interaction.search_performed else 0.0,
                float(interaction.follow_up_count),
            ]
            
            X.append(features)
            
            # Target: relevância (inferida do comportamento)
            relevance = 0.5  # Base
            
            if interaction.memory_saved:
                relevance += 0.3  # Salvou = muito relevante
            
            if interaction.result_clicked is not None:
                # Quanto mais alto clicou, mais relevante
                relevance += 0.2 / (interaction.result_clicked + 1)
            
            if interaction.copied_content:
                relevance += 0.2
            
            if interaction.follow_up_count > 0:
                relevance += min(0.2, interaction.follow_up_count * 0.05)
            
            y_relevance.append(min(1.0, relevance))
            
            # Domain classification
            domain_map = {
                "biomaterials": 0,
                "chemistry": 1,
                "research": 2,
                "psychiatry": 3,
                "internal_medicine": 4,
                "physics": 5,
                "general": 6,
            }
            y_domain.append(domain_map.get(interaction.domain, 6))
        
        return np.array(X), np.array(y_relevance), np.array(y_domain)
    
    async def _identify_expertise_areas(self, clusters: np.ndarray):
        """Identifica suas áreas de expertise vs áreas que está aprendendo"""
        
        cluster_domains = defaultdict(list)
        
        for i, cluster_id in enumerate(clusters):
            if cluster_id == -1:  # Noise
                continue
            
            interaction = self.interaction_history[i]
            cluster_domains[cluster_id].append(interaction.domain)
        
        # Áreas de expertise: muitas interações, alta profundidade
        expertise = []
        learning = []
        
        for cluster_id, domains in cluster_domains.items():
            domain_counts = Counter(domains)
            most_common_domain, count = domain_counts.most_common(1)[0]
            
            if count > 20:  # Muitas interações = expertise
                expertise.append(most_common_domain)
            elif count > 5:  # Poucas = ainda aprendendo
                learning.append(most_common_domain)
        
        self.user_profile.expertise_areas = list(set(expertise))
        self.user_profile.learning_areas = list(set(learning))
        
        logger.info(
            f"🎓 Expertise: {self.user_profile.expertise_areas} | "
            f"📚 Learning: {self.user_profile.learning_areas}"
        )
    
    async def _create_personal_embedding(self):
        """
        Cria embedding bias personalizado para você
        
        Isso faz com que buscas sejam "inclinadas" para seus interesses!
        """
        # Cria vetor de preferências baseado em domínios
        domain_list = sorted(self.user_profile.domain_preferences.keys())
        
        if not domain_list:
            return
        
        # Vetor de bias (768 dims para compatibilidade com Sentence Transformers)
        bias = np.zeros(768)
        
        # Injeta preferências nas primeiras dimensões
        for i, domain in enumerate(domain_list[:20]):  # Max 20 domínios
            preference_strength = self.user_profile.domain_preferences[domain]
            normalized = preference_strength / sum(self.user_profile.domain_preferences.values())
            
            # Distribui ao longo de múltiplas dimensões
            for j in range(i * 30, (i + 1) * 30):
                if j < 768:
                    bias[j] = normalized
        
        self.user_profile.personal_embedding_bias = bias
        logger.info("🎨 Personal embedding bias created")
    
    async def predict_relevance(self, query: str, domain: str, platform: str) -> float:
        """
        Prediz quão relevante uma query é para VOCÊ
        
        Usado para:
        - Re-ranking de resultados de busca
        - Sugestões proativas
        - Filtrar notificações
        """
        if self.preference_model is None:
            return 0.5  # Default
        
        # Cria features da query
        features = [
            self.user_profile.domain_preferences.get(domain, 0.0),
            datetime.now(timezone.utc).hour / 24.0,
            datetime.now(timezone.utc).weekday() / 7.0,
            0.5,  # Session duration (desconhecida)
            1.0 if platform == "chatgpt" else 0.0,
            1.0 if platform == "claude" else 0.0,
            1.0 if platform == "claude-code" else 0.0,
            1.0 if platform == "gemini" else 0.0,
            0.0,  # memory_saved
            0.0,  # search_performed
            0.0,  # follow_up_count
        ]
        
        relevance = self.preference_model.predict([features])[0]
        return float(np.clip(relevance, 0.0, 1.0))
    
    async def get_proactive_suggestions(self, context: Optional[str] = None) -> List[str]:
        """
        Sugestões proativas baseadas em seus padrões
        
        Ex: "Você geralmente pesquisa biomaterials às 14h, quer ver atualizações?"
        """
        suggestions = []
        
        # 1. Baseado em horário
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in self.user_profile.active_hours:
            # Sugere tópicos que você geralmente pesquisa nesse horário
            hour_interactions = [
                i for i in self.interaction_history
                if i.timestamp.hour == current_hour
            ]
            
            if hour_interactions:
                common_domains = Counter([i.domain for i in hour_interactions])
                top_domain = common_domains.most_common(1)[0][0]
                suggestions.append(
                    f"Você geralmente pesquisa {top_domain} às {current_hour}h. "
                    f"Quer ver atualizações recentes?"
                )
        
        # 2. Baseado em áreas de aprendizado
        for learning_area in self.user_profile.learning_areas:
            suggestions.append(
                f"Novas descobertas em {learning_area} podem te interessar!"
            )
        
        # 3. Conexões cross-domain que você faz
        if context:
            for (domain1, domain2), strength in self.user_profile.cross_domain_links.items():
                if context in [domain1, domain2] and strength > 3.0:
                    other_domain = domain2 if context == domain1 else domain1
                    suggestions.append(
                        f"Você frequentemente conecta {context} com {other_domain}. "
                        f"Quer explorar essa conexão?"
                    )
        
        return suggestions[:5]  # Max 5 sugestões
    
    async def personalize_search_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Re-ranqueia resultados de busca baseado em SEU perfil
        
        Resultados mais alinhados com seus interesses sobem!
        """
        if not results or self.preference_model is None:
            return results
        
        # Calcula score personalizado para cada resultado
        scored_results = []
        
        for result in results:
            # Score original (similaridade semântica)
            original_score = result.get("score", 0.5)
            
            # Score personalizado
            result_domain = result.get("domain", domain)
            result_platform = result.get("platform", "other")
            
            personal_relevance = await self.predict_relevance(
                query, result_domain, result_platform
            )
            
            # Combina scores (70% semântico + 30% personalizado)
            final_score = 0.7 * (1 - original_score) + 0.3 * personal_relevance
            
            # Boost se domínio de expertise
            if result_domain in self.user_profile.expertise_areas:
                final_score *= 1.1
            
            # Boost se conexão cross-domain conhecida
            for (d1, d2), strength in self.user_profile.cross_domain_links.items():
                if result_domain in [d1, d2] and domain in [d1, d2]:
                    final_score *= (1 + strength * 0.01)
            
            result["personalized_score"] = final_score
            scored_results.append(result)
        
        # Re-rankeia
        scored_results.sort(key=lambda x: x["personalized_score"], reverse=True)
        
        return scored_results
    
    async def export_profile(self) -> Dict[str, Any]:
        """Exporta perfil do usuário para análise/backup"""
        return {
            "user_id": self.user_profile.user_id,
            "domain_preferences": dict(self.user_profile.domain_preferences),
            "platform_by_task": dict(self.user_profile.platform_by_task),
            "active_hours": self.user_profile.active_hours,
            "expertise_areas": self.user_profile.expertise_areas,
            "learning_areas": self.user_profile.learning_areas,
            "cross_domain_links": {
                f"{k[0]}-{k[1]}": v 
                for k, v in self.user_profile.cross_domain_links.items()
            },
            "metrics": self.metrics,
            "total_interactions": len(self.interaction_history),
            "last_updated": self.user_profile.last_updated.isoformat(),
        }


# Singleton global
_continuous_learning_engine: Optional[ContinuousLearningEngine] = None


def get_continuous_learning_engine(semantic_memory_service=None) -> ContinuousLearningEngine:
    """Get or create continuous learning engine singleton"""
    global _continuous_learning_engine
    
    if _continuous_learning_engine is None:
        if semantic_memory_service is None:
            raise ValueError("semantic_memory_service required for initialization")
        
        _continuous_learning_engine = ContinuousLearningEngine(semantic_memory_service)
    
    return _continuous_learning_engine

