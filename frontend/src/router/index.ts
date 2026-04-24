// frontend/src/router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import DataUploadView from '../views/DataUploadView.vue'
import PipelineView from '../views/PipelineView.vue'
import ResultsView from '../views/ResultsView.vue'

const routes = [
  { path: '/', redirect: '/data' },
  { path: '/data', component: DataUploadView, meta: { title: 'Data Upload' } },
  { path: '/pipeline', component: PipelineView, meta: { title: 'Pipeline Config' } },
  { path: '/results', component: ResultsView, meta: { title: 'Results' } },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  if (to.meta.title) {
    document.title = `ChemAI Nexus - ${to.meta.title}`
  }
  next()
})

export default router
